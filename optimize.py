import numpy as np
from scipy.optimize import fsolve
import math

c_p = 381           # Specific heat (J/(kgK))
Tm = 302.8          # Melting point (K)
T_H = 305           # Hot wall temperature (K)
Lat = 8.016e5       # Latent heat (J/kg)

St_l = c_p * (T_H - Tm) / Lat

def equation(p):
    xi = p

    return xi * np.exp(xi**2) * math.erf(xi) - St_l / np.sqrt(np.pi)

xi = fsolve(equation, 2)
print(xi)
