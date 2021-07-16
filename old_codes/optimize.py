import numpy as np
from scipy.optimize import fsolve
import math

c_p = 381           # Specific heat (J/(kgK))
Tm = 302.8          # Melting point (K)
T_H = 322.5           # Hot wall temperature (K)
Lat = 8.016e4       # Latent heat (J/kg)

St_l = c_p * (T_H - Tm) / Lat
print(St_l)

def equation(p):
    xi = p

    return xi * np.exp(xi**2) * math.erf(xi) - St_l / np.sqrt(np.pi)

xi = fsolve(equation, 0.01)
print(xi)
