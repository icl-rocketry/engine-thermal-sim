6# Plz Delet all the titles for the text file before using this program
import numpy as np
import matplotlib.pyplot as plt
import re

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

x = np.array([25, 50]) # E Temps
y = np.array([78.9, 78.9]) * 1e9 # E
x2 = np.array([300, 400, 500, 600, 700, 800, 900, 1000]) - 273.15 # Ys Temps
y2 = np.array([175, 170, 160, 150, 135, 120, 95, 70]) * 1e6 # Ys

coefficients = np.polyfit(x, y, 2)
coefficients2 = np.polyfit(x2, y2, 2)

A = float(coefficients[0])
B = float(coefficients[1])
C = float(coefficients[2])
A2 = float(coefficients2[0])
B2 = float(coefficients2[1])
C2 = float(coefficients2[2])



a = 20e-6 # Coefficient of thermal expansion
k = 250 # Thermal conductivity
v = 0.33
pc = 35 # regen channel pressure
t_w = (10 ** -3) * 0.8
h_rib = (10 ** -3) * 1.5
channel_arc_angle = 5 # degrees

file_path = 'Thermals_Filtered.txt' # name of the text file

pos = []
rad = []
twc = []
twg = []
tc = []
sf = []
yieldstress = []
tempstress_t = []
tempstress_l = []
tempstress_p = []
von_mises = []
w_channel = []

qtotal = []
qconv = []

data_arrays = []

with open(file_path, 'r') as file:
    for line in file:
        line_array = line.split()
        line_array = [float(item) for item in line_array]
        data_arrays.append(line_array)

        T = float(line_array[7] - 273.15)
        if -273.15 < T < 750:
            E = 78.9e9
            Ys = A2 * (T ** 2) + B2 * T + C2
        else:
            E = 0
            Ys = 0
        q = float(line_array[5]) * 1000
        stress_t = (E * a * q * t_w) / (2 * (1 - v) * k)
        stress_t2 = E * a * (float(line_array[7]) - float(line_array[8]))
        stress_p = (((float(line_array[1]*1e-3) + t_w)*np.sin(np.deg2rad(channel_arc_angle))/t_w)**2 * pc * 1e5 / 2)

        qconv.append(float(line_array[3]))
        qtotal.append(float(line_array[5]))
        twg.append(float(line_array[7]))
        twc.append(float(line_array[8]))
        tc.append(float(line_array[9]))
        #print(float(line_array[6]) - float(line_array[8]))
        line_array.append(stress_t)
        
        pos.append(float(line_array[0]) / 1000)
        rad.append(float(line_array[1]) / 1000)
        
        yieldstress.append(Ys * 1e-6)
        tempstress_t.append(stress_t * 1e-6)
        tempstress_l.append(stress_t2 * 1e-6)
        tempstress_p.append(stress_p* 1e-6)
        s1 = stress_t + stress_p
        s2 = stress_t2
        von_mises.append(np.sqrt(0.5 * ((s1 - s2)**2 + (s2)**2 + (s1)**2)) * 1e-6)
        sf = np.divide(yieldstress, von_mises)


#rc_mdot = 0.199
#rc_cp = 2.57

# Calculate total heat flux
totHeatFluxInt = 0
#totHeatFluxTemp = 0
for i in range(len(pos)-1):
    dA = np.pi * (rad[i] + rad[i+1]) * np.sqrt((rad[i] - rad[i+1]) ** 2 + (pos[i+1] - pos[i]) ** 2)
    totHeatFluxInt += qconv[i] * dA
#totHeatFluxTemp = (tc[0] - tc[-1]) * rc_cp * rc_mdot
print(f"Regen Cooling Power: {totHeatFluxInt:.1f} kW")
#print('Total Heat Flux:', totHeatFluxTemp, 'kW - Temperature')


#for array in data_arrays:
#    print(array)
#print(line_array)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#ax.plot(pos, rad)
ax[0].set_ylabel("Chamber Radius (m)")
ax2 = ax[0]
ax2.plot(pos, yieldstress, color="tab:green"    , label="Yield stress")
ax2.plot(pos, tempstress_t, color="tab:pink"    , label="Tangential Thermal")
ax2.plot(pos, tempstress_l, color="tab:purple"  , label="Longitudinal Thermal")
ax2.plot(pos, tempstress_p, color="tab:orange"  , label="Tangential Pressure")
ax2.plot(pos, von_mises, color="tab:red"        , label="Von-Mises")
ax2.set_ylabel("Stress (MPA)")
ax2.set_xlabel("Axial Distance")
ax[0].grid()
ax2.legend()
ax3 = ax[1]
ax3.plot(pos, twg, label="twg")
ax3.plot(pos, twc, label='twc')
ax3.plot(pos, tc, label='tc')
ax3.set_ylabel("Temperature (K)")
ax3.set_xlabel("Axial Distance")
ax3.legend()
ax3.grid()
ax4 = ax[2]
ax4.plot(pos, sf, color="tab:red")
ax4.set_ylabel("Safety Factor")
ax4.set_xlabel("Axial Distance")
ax4.set_ylim(0, None)
ax4.grid()
min_sf = np.min(sf)
min_sf_index = np.argmin(sf)
ax4.set_title(f"Minimum Safety Factor: {min_sf:.3f}")
ax4.axhline(y=min_sf, color='black', linestyle='--')
ax4.legend(["Safety Factor", "Minimum Safety Factor"])

plt.show()