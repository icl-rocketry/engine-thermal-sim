import numpy as np
from thermo.chemical import Chemical

ipa = Chemical("isopropanol")
mw = ipa.MW

pressure = np.array([1, 10, 20, 30, 40, 50, 60]) * 1e5 # Pa
temperature = np.arange(270, 510, 10) # K

rho = np.zeros((len(pressure), len(temperature))) # kg/m3
cp = np.zeros((len(pressure), len(temperature))) # kJ/kg·K
k = np.zeros((len(pressure), len(temperature))) # mW/m·K
visc = np.zeros((len(pressure), len(temperature))) # mu Pa·s
Tsat = np.zeros(len(pressure)) # K
qvap = np.zeros_like(Tsat) # kJ/kg

for j, t in enumerate(temperature):
    for i, p in enumerate(pressure):
        ipa.calculate(t, p)
        rho[i, j] = ipa.rho
        cp[i, j] = ipa.Cp * 1e-3
        k[i, j] = ipa.k * 1e3
        visc[i, j] = ipa.mu * 1e6
        if j == 0:
            Tsat[i] = ipa.Tsat(p)
            qvap[i] = ipa.EnthalpyVaporization(Tsat[i]) / mw if ipa.EnthalpyVaporization(Tsat[i]) is not None else None

width = 14

print(f"{ipa.formula}(L)          {len(pressure)},{len(temperature)}")

for i, p in enumerate(pressure):
    for j, t in enumerate(temperature):
        p_str = f" {p/1e6:<{width-2}.6f}"
        t_str = f"{t:<{width}.2f}"
        cp_str = f"{cp[i,j]:<{width}.5f}"
        rho_str = f"{rho[i,j]:<{width}.4f}"
        mu_str = f"{visc[i,j]:<{width}.4f}"
        k_str = f"{k[i,j]:<{width}.5f}"
        
        if j == 0:
            tsat_str = f"{Tsat[i]:<{width}.2f}" if Tsat[i] < temperature[-1] else f"{temperature[-1]:<{width}.2f}"
            qvap_str = f"{qvap[i] if qvap[i] is not None else 0:<{width}.4f}" if ~np.isnan(qvap[i]) else ""
        else:
            tsat_str = ""
            qvap_str = ""

        print(f" {p_str} {t_str} {cp_str} {rho_str} {mu_str} {k_str} {tsat_str} {qvap_str}")