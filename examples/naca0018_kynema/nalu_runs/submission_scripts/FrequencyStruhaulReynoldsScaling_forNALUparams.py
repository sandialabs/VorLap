import math

# The intent of this script is to identify what the upper and lower bounds of the captured frequency content is compared to the expected range of frequencies between flat plate and cylinder.  
# We want to minimize the total computational cost while ensuring we capture the correct frequency band with enough detail.
# This was originally done by scaling the original Shreyas and Ganesh factors based on the reynolds number, but since the tuning factors were tuned, it might have been better to just start with the desired frequency band and then see if the timesteps were acceptable.

# Change These
Vinf_new = 7.5

# Frequency Tuning Parameters
N_flowthrough_factor = 0.29
N_steps_factor = 2.665

# Don't Change These
Chord_new = 1.0
Timestep_orig = 0.0002667
Steps_orig = 16000.0
Reynolds_orig = 9333333.333333
fluid_density = 1.2
fluid_viscosity = 9e-06

Cmu = 0.09
chord_len_fract = 0.07
Turbulent_Intensity = 0.01

Reynolds = fluid_density / fluid_viscosity * Chord_new * Vinf_new
Timestepw_per_linear_scaling = Timestep_orig * Reynolds_orig / Reynolds / N_flowthrough_factor
Steps = Steps_orig / N_steps_factor
total_T = Steps * Timestepw_per_linear_scaling
turbulent_ke = 3 / 2 * (Vinf_new * Turbulent_Intensity) ** 2
specific_dissipation_rate = math.sqrt(turbulent_ke) / ((Cmu ** 0.25) * chord_len_fract * Chord_new)

# Keep temporal fidelity for shedding predictions
target_st_min = 0.15
target_st_max = 0.40
target_points_per_cycle = 120
target_cycles_captured = 50
max_dt_from_frequency = Chord_new / (target_points_per_cycle * target_st_max * Vinf_new)
min_total_time_from_cycles = target_cycles_captured * Chord_new / (target_st_min * Vinf_new)
guarded_time_step = min(Timestepw_per_linear_scaling, max_dt_from_frequency)
guarded_steps = max(int(math.ceil(Steps)), int(math.ceil(min_total_time_from_cycles / guarded_time_step)))
guarded_total_T = guarded_steps * guarded_time_step

print("__________________")
print("Reynolds: ", Reynolds)
print("Timestepw_per_linear_scaling: ", Timestepw_per_linear_scaling)
print("Steps: ", Steps)
print("total_T: ", total_T)
print("guarded_time_step: ", guarded_time_step)
print("guarded_steps: ", guarded_steps)
print("guarded_total_T: ", guarded_total_T)
print("turbulent_ke: ", turbulent_ke)
print("specific_dissipation_rate: ", specific_dissipation_rate)

N_Flowthroughs = total_T * Vinf_new / Chord_new
min_Freq = 1 / (Timestepw_per_linear_scaling * Steps)
max_Freq = Steps / (Timestepw_per_linear_scaling * Steps)

Shed_Freq_of_cyl = 0.4 * Vinf_new / Chord_new
Lower_bound_factor_cyl_w_per_nyquist = Shed_Freq_of_cyl / min_Freq
Upper_Bound_Factor_cyl_w_per_nyquist = Shed_Freq_of_cyl / max_Freq
Number_of_Cyl_cycles_captured = Shed_Freq_of_cyl * total_T

Shed_Freq_of_flat_plate = 0.2 * Vinf_new / Chord_new
Number_of_plate_cycles_captured = Shed_Freq_of_flat_plate * total_T
Lower_bound_factor_plate_w_per_nyquist = Shed_Freq_of_flat_plate / min_Freq
Upper_Bound_Factor_plate_w_per_nyquist = Shed_Freq_of_flat_plate / max_Freq

print()
print("N_Flowthroughs: ", N_Flowthroughs)
print("min_Freq: ", min_Freq)
print("max_Freq: ", max_Freq)
print()
print("Shed_Freq_of_cyl: ", Shed_Freq_of_cyl)
print("Lower_bound_factor_cyl_w_per_nyquist: ", Lower_bound_factor_cyl_w_per_nyquist)
print("Upper_Bound_Factor_cyl_w_per_nyquist: ", Upper_Bound_Factor_cyl_w_per_nyquist)
print("Number_of_Cyl_cycles_captured: ", Number_of_Cyl_cycles_captured)
print()
print("Shed_Freq_of_flat_plate: ", Shed_Freq_of_flat_plate)
print("Number_of_plate_cycles_captured: ", Number_of_plate_cycles_captured)
print("Lower_bound_factor_plate_w_per_nyquist: ", Lower_bound_factor_plate_w_per_nyquist)
print("Upper_Bound_Factor_plate_w_per_nyquist: ", Upper_Bound_Factor_plate_w_per_nyquist)
