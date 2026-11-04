import yaml, sys, math
from pathlib import Path
import subprocess, shlex

def submit_case(af_name, run_type, aoa, Vinf_new, run_folder, slurm_script="runNALUsingle.slurm",
                extra_sbatch=None, dry_run=True):
    """
    Submit one case to Slurm with a descriptive job name.
    - extra_sbatch: list of extra sbatch args, e.g. ["--partition=batch","--qos=normal"]
    - dry_run: True prints the sbatch command without submitting
    Returns (job_id or None).
    """
    case_dir = Path(run_folder) / af_name / run_type / f"aoa_{aoa}"
    job_name = f"{af_name}_V{Vinf_new:g}_aoa{aoa}_{run_type}"
    cmd = ["sbatch", f"--job-name={job_name}"]

    # Optional: pass identifiers to the script as env vars (handy for logging inside the .slurm)
    env_export = f"ALL,MESH={af_name},VINF={Vinf_new:g},AOA={aoa},RUNTYPE={run_type}"
    cmd += [f"--export={env_export}"]

    if extra_sbatch:
        cmd += list(extra_sbatch)

    cmd += [slurm_script, str(case_dir)]

    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return None

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # sbatch usually prints: "Submitted batch job 123456"
    out = res.stdout.strip()
    print(out)
    # Extract job id if present
    parts = out.split()
    job_id = parts[-1] if parts and parts[-1].isdigit() else None
    return job_id


def gen_static_case(
    af_name,
    grid_file,
    aoa,
    Vinf_new,
    run_folder='../',
    slurm_script="runNALUsingle.slurm",
    template="./templates/template_nalu_run.yaml",
    dry_run=True,
    flow_direction=(1.0, 0.0, 0.0),
):

    # Don't Change These Factors unless you used the excel sheet or python script to tune your frequency content and have the desired numbers to insert.
    N_flowthrough_factor = 0.29
    N_steps_factor = 2.665
    Chord_new = 1.0
    Timestep_orig = 0.0002667
    Steps_orig = 16000.0
    Reynolds_orig = 9333333.333333
    fluid_density = 1.2
    fluid_viscosity = 9e-06
    Cmu = 0.09
    chord_len_fract = 0.07
    Turbulent_Intensity = 0.01

    # Calculations
    if Vinf_new <= 0.0:
        raise ValueError(f"Vinf_new must be > 0.0, got {Vinf_new}")
    flow_mag = math.sqrt(flow_direction[0] ** 2 + flow_direction[1] ** 2 + flow_direction[2] ** 2)
    if flow_mag <= 0.0:
        raise ValueError(f"flow_direction must have non-zero magnitude, got {flow_direction}")
    flow_dir = [flow_direction[0] / flow_mag, flow_direction[1] / flow_mag, flow_direction[2] / flow_mag]
    inflow_velocity = [Vinf_new * flow_dir[0], Vinf_new * flow_dir[1], Vinf_new * flow_dir[2]]

    Reynolds = fluid_density / fluid_viscosity * Chord_new * Vinf_new
    Timestepw_per_linear_scaling = (Timestep_orig * Reynolds_orig / Reynolds / N_flowthrough_factor)
    Steps = Steps_orig / N_steps_factor
    total_T = Steps * Timestepw_per_linear_scaling

    # Strouhal-resolution guard:
    # keep enough points/cycle and cycles captured so shedding frequency is not under-resolved.
    # This protects low-Re cases where the legacy scaling can pick very large dt values.
    target_st_min = 0.15
    target_st_max = 0.40
    target_points_per_cycle = 120
    target_cycles_captured = 50

    max_dt_from_frequency = Chord_new / (target_points_per_cycle * target_st_max * Vinf_new)
    min_total_time_from_cycles = target_cycles_captured * Chord_new / (target_st_min * Vinf_new)
    guarded_time_step = min(Timestepw_per_linear_scaling, max_dt_from_frequency)
    guarded_steps = max(int(math.ceil(Steps)), int(math.ceil(min_total_time_from_cycles / guarded_time_step)))
    total_T_guarded = guarded_steps * guarded_time_step
    turbulent_ke = 3 / 2 * (Vinf_new * Turbulent_Intensity) ** 2
    specific_dissipation_rate = math.sqrt(turbulent_ke) / ((Cmu ** 0.25) * chord_len_fract * Chord_new) 

    mant, exp = f"{round(Reynolds):.2E}".split("E")
    run_type = f"RE{mant.replace('.', '_')}E{int(exp)}" #run_type = "RE4_00E4"

    if ( not Path(template).exists() ):
        print("Template file ", template, " doesn't exist. Please check your inputs")
        sys.exit()

    if ( not Path(grid_file).exists() ):
        print("Mesh file ", grid_file, " doesn't exist. Please check your inputs")
        sys.exit()

    tfile = yaml.safe_load(Path(template).read_text())

    tfile['linear_solvers'][0]['hypre_cfg_file'] = str(Path(template).resolve().parent / 'hypre_file.yaml')
    tfile['linear_solvers'][1]['hypre_cfg_file'] = str(Path(template).resolve().parent / 'hypre_file.yaml')
    tfile['linear_solvers'][2]['hypre_cfg_file'] = str(Path(template).resolve().parent / 'hypre_file.yaml') 

    tfile['realms'][0]['mesh'] = str(Path(grid_file).absolute())

    tfile['Time_Integrators'][0]['StandardTimeIntegrator']['termination_step_count'] = guarded_steps + 1
    tfile['Time_Integrators'][0]['StandardTimeIntegrator']['time_step'] = guarded_time_step

    tfile['realms'][0]['output']['output_data_base_name'] = run_folder + '/' + af_name + '/' + run_type + '/aoa_' + str(aoa) + '/results/{}_{}.e'.format(af_name, aoa)
    tfile['realms'][0]['output']['output_frequency'] = 1000
    tfile['realms'][0]['post_processing'][0]['output_file_name'] = run_folder + '/' + af_name + '/' + run_type + '/aoa_' + str(aoa) + '/pp_'+af_name+'_'+str(aoa)+'.dat'
    tfile['realms'][0]['post_processing'][1]['output_file_name'] = run_folder + '/' + af_name + '/' + run_type + '/aoa_' + str(aoa) + '/' + af_name+'_'+str(aoa)+'.dat'
    tfile['realms'][0]['restart']['restart_data_base_name'] = run_folder + '/' + af_name + '/' + run_type + '/aoa_' + str(aoa) + '/restart/{}_{}.e'.format(af_name, aoa)
    tfile['realms'][0]['restart']['restart_frequency'] = 1000
    tfile['realms'][0]['solution_options']['turbulence_model'] = 'sst_iddes'

    # Helper closure-free walkers
    def update_block(d):
        if 'turbulent_ke' in d and d['turbulent_ke'] != 0:
            d['turbulent_ke'] = turbulent_ke
        if 'specific_dissipation_rate' in d and d['specific_dissipation_rate'] != 0:
            d['specific_dissipation_rate'] = specific_dissipation_rate
        if 'velocity' in d:
            v = d['velocity']
            if isinstance(v, list) and len(v) >= 3 and float(v[0]) == 0.3 and float(v[1]) == 0.0 and float(v[2]) == 0.0:
                d['velocity'] = inflow_velocity

    # boundary_conditions
    for bc in tfile['realms'][0]['boundary_conditions']:
        if 'inflow_user_data' in bc:
            update_block(bc['inflow_user_data'])
        if 'open_user_data' in bc:
            update_block(bc['open_user_data'])
        if 'wall_user_data' in bc:
            update_block(bc['wall_user_data'])

    # initial_conditions
    for ic in tfile['realms'][0]['initial_conditions']:
        if 'value' in ic:
            update_block(ic['value'])

    # Each preparation is additive. Never erase an earlier CFD result.
    p = Path(run_folder) / af_name / run_type / f"aoa_{aoa}"
    p.mkdir(parents=True, exist_ok=False)

    out_yaml = Path(run_folder) / af_name / run_type / f"aoa_{aoa}" / f"{af_name}_{run_type}_aoa_{aoa}.yaml"
    yaml.dump(tfile, open(out_yaml, "w"), default_flow_style=False)

    print(
        "[time-controls]",
        f"aoa={aoa}",
        f"u_inf={Vinf_new:.6g}",
        f"flow_dir=({flow_dir[0]:.5f},{flow_dir[1]:.5f},{flow_dir[2]:.5f})",
        f"Re={Reynolds:.6g}",
        f"legacy_dt={Timestepw_per_linear_scaling:.6g}",
        f"guarded_dt={guarded_time_step:.6g}",
        f"steps={guarded_steps + 1}",
        f"total_time={total_T_guarded:.6g}",
    )

    # Submit to Slurm (tweak extra_sbatch as you like)
    submit_case(af_name, run_type, aoa, Vinf_new, run_folder,
                slurm_script=slurm_script,
                dry_run=dry_run,
                extra_sbatch=[
                    "--nodes=2",
                    "--time=31:59:59",
                ])



def main():
    import argparse
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare one Kynema/Nalu case; submission is explicit.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--airfoil", default="NACA0018")
    parser.add_argument("--aoa", type=float, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.mesh.is_file():
        parser.error("mesh file does not exist")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.airfoil):
        parser.error("airfoil must be a simple identifier")
    gen_static_case(args.airfoil, str(args.mesh.resolve()), args.aoa, args.speed,
        str(args.output.resolve()), str(here/"templates/runNALUsingle.slurm"),
        str(here/"templates/template_nalu_run.yaml"), not args.submit)


if __name__ == "__main__":
    main()
