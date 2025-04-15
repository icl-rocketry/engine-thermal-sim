import numpy as np
import matplotlib.pyplot as plt
import re
from os import system

system('cls')

# All material data from Eurocode 9

ys_0 = 260e6 # Yield stress at room temp (for under 6mm thickness)
uts_0 = 310e6 # Ultimate tensile strength at room temp (for under 6mm thickness)
v = 0.3 # Poisson's ratio

modulus_temps = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 550]) # E Temps (deg C)
modulus = np.array([70, 69.3, 67.9, 65.1, 60.2, 54.6, 47.6, 37.8, 28.0, 0]) * 1e9 # E

yield_temps = np.array([20, 100, 150, 200, 250, 300, 350, 550]) # Ys Temps (deg C)
yield_stress = np.array([1, 0.90, 0.79, 0.65, 0.38, 0.20, 0.11, 0]) * ys_0 # Ys

thermal_cond = lambda T: 0.07*T + 190

cte = lambda T: 0.2e-7*T + 22.5e-6

# Engine Data

avg_channel_pressure = 40 # regen channel pressure (bar) - assumed to be constant
t_w = 1e-3 * 0.6 # wall thickness (mm)
channel_width = 1e-3 * 0.4 # mm

#############################################################################################

with open('RPA_Thermals.txt', 'r', encoding="utf8") as input_file:
    lines = input_file.readlines()

lines = lines[8:]
filtered_lines = [line for line in lines if not line.startswith('#')]

# Define a regular expression to match numbers
number_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')

filtered_lines = []
for line in lines:
    # Extract numerical values from the line using the regular expression
    numbers = number_pattern.findall(line)
    # Join the numbers into a comma-separated string and add it to the filtered lines
    filtered_line = (' '.join(numbers) + '\n')
    if filtered_line.strip():
        filtered_lines.append(filtered_line)

with open('Thermals_Filtered.txt', 'w') as output_file:
    output_file.writelines(filtered_lines)

file_path = 'Thermals_Filtered.txt' # name of the text file

# Initialize all arrays as numpy arrays
axial_pos = np.array([])
r = np.array([])
coolant_wall_temp = np.array([])
chamber_wall_temp = np.array([])
coolant_temp = np.array([])
yield_sf = np.array([])
yield_stress_array = np.array([])
modulus_array = np.array([])
tangential_thermal_stress = np.array([])
longitudinal_thermal_stress = np.array([])
pressure_stress = np.array([])
von_mises_stress = np.array([])
strain = np.array([])
q_total = np.array([])
q_conv = np.array([])
rpa_channel_pressure = np.array([])
coolant_velocity = np.array([])
coolant_density = np.array([])
crit_long_buckling_stress = np.array([])

data_arrays = []

with open(file_path, 'r') as file:
    for line in file:
        line_array = np.array([float(item) for item in line.split()])
        data_arrays.append(line_array)

        T = float(line_array[7] - 273.15) # T in deg C
        if 0 <= T <= np.min([np.max(yield_temps), np.max(modulus_temps)]):
            E =  np.interp(T, modulus_temps, modulus)
            station_yield_strength = np.interp(T, yield_temps, yield_stress)
        else:
            E = 0
            station_yield_strength = 0
        q = float(line_array[5]) * 1e3
        stress_t = (E * cte(T) * q * t_w) / (2 * (1 - v) * thermal_cond(T))
        stress_t2 = E * cte(T) * (float(line_array[7]) - float(line_array[8]))
        stress_p = (channel_width/t_w)**2 * avg_channel_pressure * 1e5 / 2

        crit_long_buckling_stress = np.append(crit_long_buckling_stress, [E * t_w / (1e-3 * float(line_array[1]) * np.sqrt(3*(1 - v**2)))])

        # Append values to numpy arrays using np.append()
        q_conv = np.append(q_conv, float(line_array[3]) * 1e3)
        q_total = np.append(q_total, q)
        chamber_wall_temp = np.append(chamber_wall_temp, float(line_array[7]))
        coolant_wall_temp = np.append(coolant_wall_temp, float(line_array[8]))
        coolant_temp = np.append(coolant_temp, float(line_array[9]))
        rpa_channel_pressure = np.append(rpa_channel_pressure, float(line_array[10])/10)
        coolant_velocity = np.append(coolant_velocity, float(line_array[11]))
        coolant_density = np.append(coolant_density, float(line_array[12]))

        axial_pos = np.append(axial_pos, float(line_array[0]) * 1e-3)
        r = np.append(r, float(line_array[1]) * 1e-3)
         
        yield_stress_array = np.append(yield_stress_array, station_yield_strength * 1e-6)
        modulus_array = np.append(modulus_array, E * 1e-9)
        tangential_thermal_stress = np.append(tangential_thermal_stress, stress_t * 1e-6)
        longitudinal_thermal_stress = np.append(longitudinal_thermal_stress, stress_t2 * 1e-6)
        pressure_stress = np.append(pressure_stress, stress_p * 1e-6)
        
        s1 = stress_t + stress_p
        s2 = stress_t2
        von_mises_temp = np.sqrt(0.5 * ((s1 - s2)**2 + (s2)**2 + (s1)**2))
        von_mises_stress = np.append(von_mises_stress, np.sqrt(0.5 * ((s1 - s2)**2 + (s2)**2 + (s1)**2)) * 1e-6)
        strain = np.append(strain, 100*von_mises_temp / E)

yield_sf = np.divide(yield_stress_array, von_mises_stress)
buckling_sf = np.divide(crit_long_buckling_stress, longitudinal_thermal_stress*1e6)

# Calculate total heat flux
totHeatFluxInt = 0
for i in range(len(axial_pos)-1):
    dA = np.pi * (r[i] + r[i+1]) * np.sqrt((r[i] - r[i+1]) ** 2 + (axial_pos[i+1] - axial_pos[i]) ** 2) # mm^2
    totHeatFluxInt += q_total[i] * dA

print(f'Total Heat Flux: {totHeatFluxInt/1e3:.1f} kW')
print(f'Coolant temperature rise: {np.max(coolant_temp) - np.min(coolant_temp):.1f} deg C')
print(f'Min coolant density: {np.min(coolant_density):.1f} kg/m^3')

throat_index = np.argmin(r)
axial_pos = (axial_pos - axial_pos[throat_index]) * 1e3

min_pos = axial_pos[0]
max_pos = axial_pos[-1]

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0,0].set_ylabel("Chamber Radius (m)")
ax7 = ax[0,0].twinx()
ax2 = ax[0,0]
ax2.plot(axial_pos, yield_stress_array, color="tab:green", label="Yield Stress")
ax2.plot(axial_pos, tangential_thermal_stress, color="tab:pink", label="Tangential Thermal Stress")
ax2.plot(axial_pos, longitudinal_thermal_stress, color="tab:purple", label="Longitudinal Thermal")
ax2.plot(axial_pos, pressure_stress, color="tab:orange", label="Tangential Pressure")
ax2.plot(axial_pos, von_mises_stress, color="tab:red", label="Von-Mises")

ax7.plot(axial_pos, modulus_array, color="tab:blue", label="Modulus")
ax7.set_ylabel("Modulus (GPa)", color="tab:blue")
ax7.set_ylim(0, None)

ax2.set_ylabel("Stress (MPa)")
ax2.set_xlabel("Axial Distance (mm)")
ax2.set_xlim(min_pos, max_pos)
ax[0,0].grid()

legend_lines = ax2.lines + ax7.lines
legend_labels = ["Yield Stress", "Tangential Thermal Stress", "Longitudinal Thermal Stress", "Tangential Pressure Stress", "Von-Mises Stress", "Modulus"]
ax2.legend(legend_lines, legend_labels, loc='best')

ax3 = ax[1,0]
ax3.plot(axial_pos, chamber_wall_temp, label="twg", color="tab:orange")
ax3.plot(axial_pos, coolant_wall_temp, label='twc', color="tab:blue")
ax3.plot(axial_pos, coolant_temp, label='tc', color="tab:green")

ax8 = ax3.twinx()
ax8.plot(axial_pos, q_conv, color="tab:red", label="Heat Flux")
ax8.set_ylabel("Heat Flux (W/m^2)", color="tab:red")

ax3.set_ylabel("Temperature (K)")
ax3.set_xlabel("Axial Distance (mm)")
ax3.set_xlim(min_pos, max_pos)
ax3.grid()

legend_lines = ax3.lines + ax8.lines
legend_labels = ["twg", "twc", "tc", "Heat Flux"]
ax3.legend(legend_lines, legend_labels, loc='best')

ax4 = ax[0,1]
ax4.plot(axial_pos, yield_sf, color="tab:red")
# ax4.plot(axial_pos, buckling_sf, color="tab:blue")
ax4.set_ylabel("Safety Factor")
ax4.set_xlabel("Axial Distance (mm)")
ax4.set_ylim(0, None)
ax4.set_xlim(min_pos, max_pos)
ax4.grid()
min_sf = np.min(yield_sf)
min_sf_index = np.argmin(yield_sf)
ax4.set_title(f"Minimum Safety Factor: {min_sf:.3f}")
ax4.axhline(y=min_sf, color='black', linestyle='--')
ax4.legend(["Safety Factor (Yield)", "Minimum Safety Factor"]) #, "Safety Factor (Buckling)", "Minimum Safety Factor"])

ax5 = ax[1,1]
ax5.plot(axial_pos, strain, color="tab:blue")
ax5.set_ylabel("Strain (%)")
ax5.set_xlabel("Axial Distance (mm)")
ax5.set_xlim(min_pos, max_pos)
ax5.grid()
ax5.legend(["Strain"])

plt.tight_layout()
plt.show()