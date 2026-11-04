#!/usr/bin/env julia

using DelimitedFiles
using LinearAlgebra
import OWENS
import GXBeam

# This is the OWENS AutoCampbellDiagram GX branch extended to retain QBlade
# point masses/inertias and to track only positive-frequency physical modes.

function option(name; default=nothing)
    index = findfirst(==(name), ARGS)
    if isnothing(index)
        isnothing(default) && error("Missing required option $name")
        return default
    end
    index == length(ARGS) && error("Option $name requires a value")
    return ARGS[index + 1]
end

parse_list(value, ::Type{T}=Float64) where {T} = T[parse(T, strip(item)) for item in split(value, ',') if !isempty(strip(item))]

function numeric_csv(path)
    data, _ = readdlm(path, ',', Float64; header=true)
    return ndims(data) == 1 ? reshape(data, 1, :) : data
end

function rowmajor(values, nrow, ncol)
    return permutedims(reshape(collect(values), ncol, nrow))
end

function transform6(rotation)
    transform = zeros(eltype(rotation), 6, 6)
    transform[1:3, 1:3] .= rotation
    transform[4:6, 4:6] .= rotation
    return transform
end

function real_quadratic(vector, matrix)
    value = dot(vector, matrix * vector)
    abs(imag(value)) <= 1.0e-9 * max(1.0, abs(real(value))) ||
        error("Hermitian quadratic form has a material imaginary part: $value")
    return real(value)
end

function mode_energy_terms(eigenstate, steady_state, assembly, point_masses, component_ids)
    distributed_velocity_mass = 0.0
    point_velocity_mass = 0.0
    elastic_stiffness = 0.0
    rayleigh_damping = 0.0
    distributed_velocity_mass_by_element = zeros(Float64, length(assembly.elements))
    elastic_stiffness_by_element = zeros(Float64, length(assembly.elements))
    elastic_stiffness_by_component = Dict{Int, Float64}()
    rayleigh_damping_by_component = Dict{Int, Float64}()

    for ielem in eachindex(assembly.elements)
        element = assembly.elements[ielem]
        # GXBeam's dynamic residual uses CtCab*M*CtCab' for both the
        # translational and rotational blocks.  Use the same steady-state
        # transformation and the velocity entries extracted from the actual
        # first-order eigenvector; do not infer velocity as lambda*u.
        CtCab = GXBeam.get_C(steady_state.elements[ielem].theta)' * element.Cab
        mass_global = transform6(CtCab) * element.mass * transform6(CtCab)'
        velocity = vcat(eigenstate.elements[ielem].V, eigenstate.elements[ielem].Omega)
        element_velocity_mass = element.L * real_quadratic(velocity, mass_global)
        distributed_velocity_mass += element_velocity_mass
        distributed_velocity_mass_by_element[ielem] = element_velocity_mass

        internal_load = vcat(eigenstate.elements[ielem].Fi, eigenstate.elements[ielem].Mi)
        twice_strain_energy = element.L * real_quadratic(internal_load, element.compliance)
        elastic_stiffness += twice_strain_energy
        elastic_stiffness_by_element[ielem] = twice_strain_energy
        component_id = component_ids[ielem]
        elastic_stiffness_by_component[component_id] =
            get(elastic_stiffness_by_component, component_id, 0.0) + twice_strain_energy

        beta = element.mu[1]
        maximum(abs.(element.mu .- beta)) <= 1.0e-14 ||
            error("Energy-projected damping currently requires one stiffness-proportional beta per element")
        rayleigh_damping += beta * twice_strain_energy
        rayleigh_damping_by_component[component_id] =
            get(rayleigh_damping_by_component, component_id, 0.0) + beta * twice_strain_energy
    end

    for (node, point_mass) in point_masses
        rotation = GXBeam.get_C(steady_state.points[node].theta)
        mass_global = transform6(rotation') * point_mass.mass * transform6(rotation)
        velocity = vcat(eigenstate.points[node].V, eigenstate.points[node].Omega)
        point_velocity_mass += real_quadratic(velocity, mass_global)
    end

    return (;
        distributed_velocity_mass,
        point_velocity_mass,
        velocity_mass=distributed_velocity_mass + point_velocity_mass,
        elastic_stiffness,
        rayleigh_damping,
        elastic_stiffness_by_component,
        rayleigh_damping_by_component,
        distributed_velocity_mass_by_element,
        elastic_stiffness_by_element,
    )
end

function physical_mass_cross(state_a, state_b, steady_state, assembly, point_masses)
    cross = 0.0 + 0.0im
    for ielem in eachindex(assembly.elements)
        element = assembly.elements[ielem]
        CtCab = GXBeam.get_C(steady_state.elements[ielem].theta)' * element.Cab
        mass_global = transform6(CtCab) * element.mass * transform6(CtCab)'
        velocity_a = vcat(state_a.elements[ielem].V, state_a.elements[ielem].Omega)
        velocity_b = vcat(state_b.elements[ielem].V, state_b.elements[ielem].Omega)
        cross += element.L * dot(velocity_a, mass_global * velocity_b)
    end
    for (node, point_mass) in point_masses
        rotation = GXBeam.get_C(steady_state.points[node].theta)
        mass_global = transform6(rotation') * point_mass.mass * transform6(rotation)
        velocity_a = vcat(state_a.points[node].V, state_a.points[node].Omega)
        velocity_b = vcat(state_b.points[node].V, state_b.points[node].Omega)
        cross += dot(velocity_a, mass_global * velocity_b)
    end
    return cross
end

function degenerate_clusters(frequencies; atol=1.0e-4, rtol=1.0e-4)
    clusters = Vector{Vector{Int}}()
    for mode in eachindex(frequencies)
        if isempty(clusters)
            push!(clusters, [mode])
            continue
        end
        previous = last(last(clusters))
        tolerance = max(atol, rtol * max(abs(frequencies[mode]), abs(frequencies[previous])))
        if abs(frequencies[mode] - frequencies[previous]) <= tolerance
            push!(last(clusters), mode)
        else
            push!(clusters, [mode])
        end
    end
    return clusters
end

function main()
model_dir = abspath(option("--model-dir"))
rpms = parse_list(option("--rpms"; default="0,1,3,5"))
nmode = parse(Int, option("--modes"; default="12"))
save_rpms = parse_list(option("--save-rpms"; default="0,1"))
save_modes = parse_list(option("--save-modes"; default="1,2,3,4,5,6,7,8,9,10,11,12"), Int)
vtk_modes = parse_list(option("--vtk-modes"; default="6,12"), Int)
mode_scaling = parse(Float64, option("--mode-scaling"; default="3"))
vtk_steps = parse(Int, option("--vtk-steps"; default="36"))
gravity_on = parse(Int, option("--gravity"; default="1")) == 1

nodes_data = numeric_csv(joinpath(model_dir, "nodes.csv"))
elements_data = numeric_csv(joinpath(model_dir, "elements.csv"))
point_mass_data = numeric_csv(joinpath(model_dir, "point_masses.csv"))
fixed_data = numeric_csv(joinpath(model_dir, "fixed_nodes.csv"))
visual_data = numeric_csv(joinpath(model_dir, "visual_sections.csv"))

points = [[row[2], row[3], row[4]] for row in eachrow(nodes_data)]
start = Int.(elements_data[:, 2])
stop = Int.(elements_data[:, 3])
lengths = elements_data[:, 4]
component_ids = Int.(elements_data[:, 5])
kind_ids = Int.(elements_data[:, 6])
frames = [rowmajor(row[7:15], 3, 3) for row in eachrow(elements_data)]
stiffness = [Symmetric(rowmajor(row[16:51], 6, 6)) for row in eachrow(elements_data)]
mass = [Symmetric(rowmajor(row[52:87], 6, 6)) for row in eachrow(elements_data)]
damping = [collect(row[88:93]) for row in eachrow(elements_data)]

assembly = GXBeam.Assembly(
    points,
    start,
    stop;
    stiffness,
    mass,
    damping,
    frames,
    lengths,
)
system = GXBeam.DynamicSystem(assembly)

point_masses = Dict{Int, GXBeam.PointMass{Float64}}()
for row in eachrow(point_mass_data)
    point_masses[Int(row[1])] = GXBeam.PointMass(Symmetric(rowmajor(row[2:37], 6, 6)))
end

prescribed_conditions = Dict{Int, GXBeam.PrescribedConditions{Float64}}()
for row in eachrow(fixed_data)
    all(Int.(row[2:7]) .== 1) || error("The current runner requires all six fixed-node DOFs")
    prescribed_conditions[Int(row[1])] = GXBeam.PrescribedConditions(
        ux=0.0,
        uy=0.0,
        uz=0.0,
        theta_x=0.0,
        theta_y=0.0,
        theta_z=0.0,
    )
end

ncross = 17
sections = zeros(Float64, 3, ncross, length(points))
angles = range(0.0, 2pi; length=ncross)
for row in eachrow(visual_data)
    inode = Int(row[1])
    frame = rowmajor(row[2:10], 3, 3)
    half_y = row[11]
    half_z = row[12]
    for (index, angle) in enumerate(angles)
        sections[:, index, inode] = frame[:, 2] * (half_y * cos(angle)) + frame[:, 3] * (half_z * sin(angle))
    end
end

vtk_dir = joinpath(model_dir, "paraview")
mkpath(vtk_dir)
GXBeam.write_vtk(
    joinpath(vtk_dir, "converted_qblade_geometry"),
    assembly;
    sections,
    metadata=Dict("component_id" => component_ids, "connector_kind_id" => kind_ids),
)

frequency = zeros(length(rpms), nmode)
all_lambda = Vector{Vector{ComplexF64}}(undef, length(rpms))
all_left_vectors = Vector{Matrix{ComplexF64}}(undef, length(rpms))
all_vectors = Vector{Matrix{ComplexF64}}(undef, length(rpms))
all_states = Vector{GXBeam.AssemblyState}(undef, length(rpms))
all_mode_indices = Vector{Vector{Int}}(undef, length(rpms))
all_corruption = Vector{Vector{Float64}}(undef, length(rpms))
all_eigen_residuals = Vector{Vector{Float64}}(undef, length(rpms))
all_biorthogonality = Vector{Matrix{ComplexF64}}(undef, length(rpms))
previous_positive_left = nothing
reference_clusters = Vector{Vector{Int}}()
gravity = gravity_on ? [0.0, 0.0, -9.80665] : zeros(3)

for (irpm, rpm) in enumerate(rpms)
    println("OWENS/GXBeam Campbell point $irpm/$(length(rpms)): $rpm RPM")
    angular_velocity = [0.0, 0.0, rpm * 2pi / 60.0]
    system, lambda, left_vectors, right_vectors, converged = GXBeam.eigenvalue_analysis!(
        system,
        assembly;
        prescribed_conditions,
        point_masses,
        angular_velocity,
        gravity,
        nev=2nmode,
        eigenvector_sensitivities=false,
        left=true,
        steady_state=true,
        structural_damping=false,
    )
    converged || error("GXBeam steady/eigen solution did not converge at $rpm RPM")
    positive_indices = findall(value -> imag(value) > 0.0, lambda)
    length(positive_indices) == nmode || error(
        "Expected $nmode positive-frequency eigenvalues at $rpm RPM, found $(length(positive_indices))"
    )
    if isnothing(previous_positive_left)
        permutation = sortperm(abs.(imag.(lambda[positive_indices])))
        corruption = zeros(nmode)
    else
        correlation = previous_positive_left * system.M * right_vectors[:, positive_indices]
        permutation, corruption = GXBeam.correlate_eigenmodes(correlation)
    end
    mode_indices = positive_indices[permutation]
    if isempty(reference_clusters)
        reference_clusters = degenerate_clusters(abs.(imag.(lambda[mode_indices])) ./ (2pi))
    end
    for cluster in reference_clusters
        if length(cluster) > 1
            mode_indices[cluster] = sort(
                mode_indices[cluster]; by=index -> abs(imag(lambda[index]))
            )
            # Eigenvectors inside an exactly/nearly degenerate reference
            # subspace are not unique.  Frequency-order those branches and do
            # not report an individual-vector corruption score for them.
            corruption[cluster] .= NaN
        end
    end
    previous_positive_left = copy(left_vectors[mode_indices, :])
    frequency[irpm, :] = abs.(imag.(lambda[mode_indices])) ./ (2pi)
    all_lambda[irpm] = ComplexF64.(lambda)
    all_left_vectors[irpm] = ComplexF64.(left_vectors)
    all_vectors[irpm] = ComplexF64.(right_vectors)
    all_states[irpm] = GXBeam.AssemblyState(system, assembly; prescribed_conditions)
    all_mode_indices[irpm] = mode_indices
    all_corruption[irpm] = Float64.(corruption)
    all_eigen_residuals[irpm] = [
        begin
            vector = right_vectors[:, index]
            stiffness_vector = system.K * vector
            mass_vector = system.M * vector
            denominator = norm(stiffness_vector) + abs(lambda[index]) * norm(mass_vector)
            iszero(denominator) ? 0.0 : norm(stiffness_vector + lambda[index] * mass_vector) / denominator
        end
        for index in mode_indices
    ]
    all_biorthogonality[irpm] = ComplexF64.(
        left_vectors[mode_indices, :] * system.M * right_vectors[:, mode_indices]
    )
end

open(joinpath(model_dir, "campbell_frequencies.csv"), "w") do stream
    println(stream, join(["rpm"; ["mode_$(lpad(mode, 2, '0'))_hz" for mode in 1:nmode]], ','))
    for index in eachindex(rpms)
        println(stream, join([rpms[index]; frequency[index, :]], ','))
    end
end

open(joinpath(model_dir, "campbell_eigenvalues.csv"), "w") do stream
    println(stream, "rpm,eigen_index,tracked_mode,real_per_s,imag_per_s,frequency_hz,damping_ratio")
    for (irpm, rpm) in enumerate(rpms)
        for (index, lambda) in enumerate(all_lambda[irpm])
            tracked_mode = something(findfirst(==(index), all_mode_indices[irpm]), 0)
            damping_ratio = iszero(abs(lambda)) ? 0.0 : -real(lambda) / abs(lambda)
            println(stream, join((rpm, index, tracked_mode, real(lambda), imag(lambda), abs(imag(lambda)) / (2pi), damping_ratio), ','))
        end
    end
end

open(joinpath(model_dir, "campbell_mode_tracking.csv"), "w") do stream
    println(stream, "rpm,tracked_mode,reference_cluster,eigen_index,frequency_hz,corruption_index")
    cluster_for_mode = zeros(Int, nmode)
    for (cluster_id, cluster) in enumerate(reference_clusters)
        cluster_for_mode[cluster] .= cluster_id
    end
    for (irpm, rpm) in enumerate(rpms)
        for mode in 1:nmode
            println(stream, join((rpm, mode, cluster_for_mode[mode], all_mode_indices[irpm][mode], frequency[irpm, mode], all_corruption[irpm][mode]), ','))
        end
    end
end

open(joinpath(model_dir, "campbell_mode_shapes.csv"), "w") do stream
    columns = [
        "rpm", "tracked_mode", "eigen_index", "node_id", "x_m", "y_m", "z_m",
        "ux_real", "ux_imag", "uy_real", "uy_imag", "uz_real", "uz_imag",
        "theta_x_real", "theta_x_imag", "theta_y_real", "theta_y_imag", "theta_z_real", "theta_z_imag",
        "velocity_x_real", "velocity_x_imag", "velocity_y_real", "velocity_y_imag", "velocity_z_real", "velocity_z_imag",
        "omega_x_real", "omega_x_imag", "omega_y_real", "omega_y_imag", "omega_z_real", "omega_z_imag",
    ]
    println(stream, join(columns, ','))
    for requested_rpm in save_rpms
        irpm = argmin(abs.(rpms .- requested_rpm))
        isapprox(rpms[irpm], requested_rpm; atol=1e-9) ||
            error("Requested mode-shape RPM $requested_rpm is not in the Campbell RPM list")
        for mode in save_modes
            1 <= mode <= nmode || error("Requested mode shape $mode is outside 1:$nmode")
            eigen_index = all_mode_indices[irpm][mode]
            eigenstate = GXBeam.AssemblyState(
                all_vectors[irpm][:, eigen_index],
                system,
                assembly;
                prescribed_conditions,
            )
            for node in eachindex(points)
                state = eigenstate.points[node]
                values = ComplexF64[
                    state.u...,
                    state.theta...,
                    state.V...,
                    state.Omega...,
                ]
                scalar_values = reduce(vcat, ([real(value), imag(value)] for value in values))
                println(stream, join((rpms[irpm], mode, eigen_index, node, points[node]..., scalar_values...), ','))
            end
        end
    end
end

open(joinpath(model_dir, "campbell_mode_properties.csv"), "w") do stream
    println(
        stream,
        "rpm,tracked_mode,eigen_index,frequency_hz,eigen_residual_relative," *
        "left_right_mass_real,left_right_mass_imag,distributed_velocity_mass_raw,point_velocity_mass_raw," *
        "velocity_mass_raw,elastic_stiffness_raw,elements_csv_beta_projection_raw_unselected," *
        "raw_modal_mass_velocity_over_omega2,stiffness_to_inertia_ratio,elements_csv_beta_projected_damping_ratio_unselected",
    )
    for requested_rpm in save_rpms
        irpm = argmin(abs.(rpms .- requested_rpm))
        for mode in save_modes
            eigen_index = all_mode_indices[irpm][mode]
            eigenstate = GXBeam.AssemblyState(
                all_vectors[irpm][:, eigen_index],
                system,
                assembly;
                prescribed_conditions,
            )
            energy = mode_energy_terms(
                eigenstate, all_states[irpm], assembly, point_masses, component_ids
            )
            omega = abs(imag(all_lambda[irpm][eigen_index]))
            raw_modal_mass = energy.velocity_mass / omega^2
            stiffness_to_inertia = energy.elastic_stiffness / energy.velocity_mass
            projected_damping = energy.rayleigh_damping * omega / (2 * energy.velocity_mass)
            left_right_mass = all_biorthogonality[irpm][mode, mode]
            println(
                stream,
                join(
                    (
                        rpms[irpm], mode, eigen_index, frequency[irpm, mode], all_eigen_residuals[irpm][mode],
                        real(left_right_mass), imag(left_right_mass), energy.distributed_velocity_mass,
                        energy.point_velocity_mass, energy.velocity_mass, energy.elastic_stiffness,
                        energy.rayleigh_damping, raw_modal_mass, stiffness_to_inertia, projected_damping,
                    ),
                    ',',
                ),
            )
        end
    end
end

open(joinpath(model_dir, "campbell_mode_energy_by_component.csv"), "w") do stream
    println(
        stream,
        "rpm,tracked_mode,eigen_index,component_id,elastic_stiffness_raw," *
        "elastic_stiffness_fraction,elements_csv_beta_projection_raw_unselected," *
        "elements_csv_effective_beta_s_unselected,elements_csv_beta_projected_damping_ratio_contribution_unselected",
    )
    for requested_rpm in save_rpms
        irpm = argmin(abs.(rpms .- requested_rpm))
        for mode in save_modes
            eigen_index = all_mode_indices[irpm][mode]
            eigenstate = GXBeam.AssemblyState(
                all_vectors[irpm][:, eigen_index],
                system,
                assembly;
                prescribed_conditions,
            )
            energy = mode_energy_terms(
                eigenstate, all_states[irpm], assembly, point_masses, component_ids
            )
            omega = abs(imag(all_lambda[irpm][eigen_index]))
            for component_id in sort(collect(keys(energy.elastic_stiffness_by_component)))
                elastic = energy.elastic_stiffness_by_component[component_id]
                rayleigh = energy.rayleigh_damping_by_component[component_id]
                elastic_fraction = elastic / energy.elastic_stiffness
                effective_beta = iszero(elastic) ? 0.0 : rayleigh / elastic
                damping_contribution = rayleigh * omega / (2 * energy.velocity_mass)
                println(
                    stream,
                    join(
                        (
                            rpms[irpm], mode, eigen_index, component_id, elastic,
                            elastic_fraction, rayleigh, effective_beta, damping_contribution,
                        ),
                        ',',
                    ),
                )
            end
        end
    end
end

open(joinpath(model_dir, "campbell_mode_energy_by_element.csv"), "w") do stream
    println(
        stream,
        "rpm,tracked_mode,eigen_index,element_id,component_id,kind_id," *
        "distributed_velocity_mass_raw,distributed_velocity_mass_fraction_of_total," *
        "elastic_stiffness_raw,elastic_stiffness_fraction",
    )
    for requested_rpm in save_rpms
        irpm = argmin(abs.(rpms .- requested_rpm))
        for mode in save_modes
            eigen_index = all_mode_indices[irpm][mode]
            eigenstate = GXBeam.AssemblyState(
                all_vectors[irpm][:, eigen_index],
                system,
                assembly;
                prescribed_conditions,
            )
            energy = mode_energy_terms(
                eigenstate, all_states[irpm], assembly, point_masses, component_ids
            )
            for element_id in eachindex(assembly.elements)
                velocity_mass = energy.distributed_velocity_mass_by_element[element_id]
                elastic = energy.elastic_stiffness_by_element[element_id]
                println(
                    stream,
                    join(
                        (
                            rpms[irpm], mode, eigen_index, element_id,
                            component_ids[element_id], kind_ids[element_id], velocity_mass,
                            velocity_mass / energy.velocity_mass, elastic,
                            elastic / energy.elastic_stiffness,
                        ),
                        ',',
                    ),
                )
            end
        end
    end
end

open(joinpath(model_dir, "campbell_mode_orthogonality.csv"), "w") do stream
    println(
        stream,
        "rpm,mode_a,mode_b,system_left_mass_right_real,system_left_mass_right_imag," *
        "physical_velocity_mass_cross_real,physical_velocity_mass_cross_imag,physical_velocity_mass_mac",
    )
    for requested_rpm in save_rpms
        irpm = argmin(abs.(rpms .- requested_rpm))
        states = Dict(
            mode => GXBeam.AssemblyState(
                all_vectors[irpm][:, all_mode_indices[irpm][mode]],
                system,
                assembly;
                prescribed_conditions,
            )
            for mode in save_modes
        )
        diagonals = Dict(
            mode => real(physical_mass_cross(states[mode], states[mode], all_states[irpm], assembly, point_masses))
            for mode in save_modes
        )
        for mode_a in save_modes, mode_b in save_modes
            system_cross = all_biorthogonality[irpm][mode_a, mode_b]
            physical_cross = physical_mass_cross(
                states[mode_a], states[mode_b], all_states[irpm], assembly, point_masses
            )
            mac = abs2(physical_cross) / (diagonals[mode_a] * diagonals[mode_b])
            println(
                stream,
                join(
                    (
                        rpms[irpm], mode_a, mode_b, real(system_cross), imag(system_cross),
                        real(physical_cross), imag(physical_cross), mac,
                    ),
                    ',',
                ),
            )
        end
    end
end

for requested_rpm in save_rpms
    irpm = argmin(abs.(rpms .- requested_rpm))
    isapprox(rpms[irpm], requested_rpm; atol=1e-9) || error("Requested VTK RPM $requested_rpm is not in the Campbell RPM list")
    for mode in vtk_modes
        1 <= mode <= nmode || error("Requested VTK mode $mode is outside 1:$nmode")
        eigen_index = all_mode_indices[irpm][mode]
        eigenstate = GXBeam.AssemblyState(
            all_vectors[irpm][:, eigen_index],
            system,
            assembly;
            prescribed_conditions,
        )
        rpm_tag = replace(string(requested_rpm), "." => "p")
        name = joinpath(vtk_dir, "qblade_RPM$(rpm_tag)_mode$(lpad(mode, 2, '0'))")
        GXBeam.write_vtk(
            name,
            assembly,
            all_states[irpm],
            all_lambda[irpm][eigen_index],
            eigenstate;
            sections,
            mode_scaling,
            steps=vtk_steps,
        )
    end
end

open(joinpath(model_dir, "owens_run_metadata.txt"), "w") do stream
    degenerate_labels = [join(cluster, '-') for cluster in reference_clusters if length(cluster) > 1]
    finite_corruption = filter(isfinite, reduce(vcat, all_corruption))
    println(stream, "OWENS module: OWENS")
    println(stream, "GXBeam module: GXBeam")
    println(stream, "Julia version: ", VERSION)
    println(stream, "RPM values: ", join(rpms, ','))
    println(stream, "Modes: ", nmode)
    println(stream, "Gravity: ", gravity_on)
    println(stream, "Point-mass nodes: ", length(point_masses))
    println(stream, "Mode tracking: positive-frequency left/right mass-matrix correlation")
    println(stream, "Degenerate zero-RPM mode clusters (frequency ordered): ", join(degenerate_labels, ';'))
    println(stream, "Maximum nondegenerate mode-tracking corruption index: ", maximum(finite_corruption))
    println(stream, "Structural damping: false (matches OWENS AutoCampbellDiagram GX default)")
    println(stream, "elements.csv beta storage: qblade_2_0_9_7_documented candidate, unselected and unused by the eigensolve")
    println(stream, "Saved mode state: GXBeam first-order mixed right eigenvector postprocessed into point u/theta/V/Omega and element Fi/Mi")
    println(stream, "Modal mass energy: exact converted element and point mass matrices projected with saved eigenvector velocity states")
    println(stream, "Damping projection basis: per-element undamped modal strain energy; candidate QBlade beta semantics are resolved only in postprocessing and numerical eigenvalue real parts are not used")
end

println("Campbell frequencies written to ", joinpath(model_dir, "campbell_frequencies.csv"))
println("ParaView geometry and modes written to ", vtk_dir)
end

main()
