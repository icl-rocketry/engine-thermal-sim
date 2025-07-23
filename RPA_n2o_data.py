import numpy as np
from thermo.chemical import Chemical

n2o = Chemical("nitrous oxide")
mw = n2o.MW

pressure = np.array([1, 10, 20, 30, 40, 50, 60, 70]) * 1e5 # Pa
temperature = np.arange(210, 510, 10) # K

rho = np.zeros((len(pressure), len(temperature))) # kg/m3
cp = np.zeros((len(pressure), len(temperature))) # kJ/kg·K
k = np.zeros((len(pressure), len(temperature))) # mW/m·K
visc = np.zeros((len(pressure), len(temperature))) # mu Pa·s
Tsat = np.zeros(len(pressure)) # K
qvap = np.zeros_like(Tsat) # kJ/kg

for j, t in enumerate(temperature):
    for i, p in enumerate(pressure):
        n2o.calculate(t, p)
        rho[i, j] = n2o.rho
        cp[i, j] = n2o.Cp * 1e-3
        k[i, j] = n2o.k * 1e3
        visc[i, j] = n2o.mu * 1e6
        if j == 0:
            Tsat[i] = n2o.Tsat(p)
            qvap[i] = n2o.EnthalpyVaporization(Tsat[i]) / mw if n2o.EnthalpyVaporization(Tsat[i]) is not None else None

width = 14

print(f"{n2o.formula}(L)         {len(pressure)},{len(temperature)}")

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