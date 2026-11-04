# Coordinate Frames and Sign Conventions

VorLap uses explicit chord, normal, span, and global bases. This page records the
current implementation conventions for offline runs, QBlade loading files, and
QBlade SIL coupling.

## Global and Component Coordinates

- Global vectors are three-component Cartesian arrays in meters, meters per
  second, newtons, or newton-meters depending on context.
- `Component.shape_xyz` stores component node positions in the component/source
  frame.
- `Component.shape_xyz_global` stores positions after frame preparation. When a
  component does not provide a separate global shape, `prepare_component_frames()`
  uses `shape_xyz`.
- `Component.chord_vector` and `Component.normal_vector` are nodewise vectors.
  They are normalized before force assembly.
- A span direction is derived from adjacent node positions for geometry-audit
  checks and nodal span weighting.

## Rotation and Inflow

`VIV_Params.rotation_axis` defines the axis used by `rotate_vector()` and point
rotation. `VIV_Params.rotation_axis_offset` is the moment reference point for
offline global moments.

For an azimuth sample:

1. The chord and normal unit vectors are rotated about `rotation_axis`.
2. The inflow vector is normalized from `VIV_Params.inflow_vec` and multiplied by
   the current inflow speed.
3. Local components are computed as dot products:
   `V_chord = dot(Vinf, chord_rotated)` and
   `V_normal = dot(Vinf, normal_rotated)`.
4. Angle of attack is `atan2(V_normal, V_chord)` in radians, converted to
   degrees for airfoil-source lookup.
5. Effective speed is `sqrt(V_chord^2 + V_normal^2)`.

The current force path treats HDF5 `CL` as the normal-direction coefficient and
`CD` as the chord-direction coefficient for bridge force assembly. This naming
is kept for file compatibility, but code and papers should describe the bridge
implementation as normal/chord coefficient reconstruction unless a future
flow-aligned lift/drag transform is added.

## Force and Moment Assembly

For each node, dynamic pressure scaling uses:

`q_node = 0.5 * rho * V_eff^2 * chord * local_segment_length`

Mean force terms use spectral index 0 when `include_mean_load=True`. QBlade
bridge preparation defaults to `include_mean_load=False` so QBlade's own mean
aerodynamics are not double-counted by VorLap. Offline analysis keeps mean load
enabled by default.

The assembled force is:

`F_global = C_chord * q_node * chord_rotated + C_normal * q_node * normal_rotated`

where current file fields map as `C_chord = CD` and `C_normal = CL`.

Offline moments are physical cross products:

`M_global = (rotated_position - rotation_axis_offset) x F_global`

The live QBlade SIL bridge currently writes force actions only. Moment spectra
are loaded and validated as source metadata, but they are not written to QBlade
torque columns by the active force path.

## QBlade Node and Action Conventions

Supported bridge node ids are:

- `BLD_<blade>_<normalized_position>`
- `STR_<strut>_<blade>_<normalized_position>`
- `TWR_<section>_<normalized_position>`
- `TWR_<normalized_position>` as a short tower form

Normalized positions must be finite and within `[0, 1]`. Duplicate structural
node ids are validation errors before case preparation or run start.

QBlade force rows use `ADDFORCE` with an action id and normalized position
derived from the node id. The generated external-library table uses global
velocity input rows named like `X_g Vel. BLD_1 pos 0.047 [m/s]` and global force
output rows for X, Y, and Z. Generated table rows set the QBlade local/global
flag to `False`, meaning global-force actions. Diagnostic constant-force probes
can use either global or local actions through their explicit `LOCAL` flag.

## Worked One-Node Example

Assume:

- `rotation_axis = [0, 0, 1]`
- `rotation_axis_offset = [0, 0, 0]`
- `inflow_vec = [1, 0, 0]`
- azimuth `0 deg`
- `chord_vector = [1, 0, 0]`
- `normal_vector = [0, 1, 0]`
- inflow speed `10 m/s`

Then:

- `V_chord = 10 m/s`
- `V_normal = 0 m/s`
- `AOA = atan2(0, 10) = 0 deg`
- `V_eff = 10 m/s`

If `CD = 0.2`, `CL = 0.5`, and `q_node = 12 N`, the global nodal force is:

- chord component: `0.2 * 12 * [1, 0, 0] = [2.4, 0, 0] N`
- normal component: `0.5 * 12 * [0, 1, 0] = [0, 6.0, 0] N`
- total: `[2.4, 6.0, 0] N`

At azimuth `90 deg` about the same positive Z axis, the chord and normal vectors
rotate with the component before the dot products and force assembly are
evaluated.

## Known Evidence Gaps

The pre-run geometry audit records VorLap-side coordinates and basis status, but
it is not yet a direct proof against QBlade-reported global/deformed node
positions. A post-run comparison against exported QBlade position channels needs
the correct frame, azimuth, and deformation assumptions before it can be used as
quantitative evidence.
