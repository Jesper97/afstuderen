import numpy as np
from scipy.optimize import least_squares, fsolve
import math

sqrt = np.emath.sqrt

Time = 400          # (s)
L = 0.1             # Length of cavity (m)
H = 0.714*L         # Height of cavity (m)
g = 9.81            # Gravitational acceleration (m/s^2)
rho0 = 6.093e3      # Density (kg/m^3)
lbda = 33           # Thermal conductivity (W/m K)
mu = 1.81e-3        # Dynamic viscosity (Ns/m^2)
nu = mu / rho0      # Kinematic viscosity (m^2/s)
beta = 1.2e-4       # Thermal expansion (1/K)
Lat = 8.016e5       # Latent heat (J/kg)
c_p = 381           # Specific heat (J/(kgK))
alpha = lbda / (rho0 * c_p)     # Thermal diffusivity (m^2/s)
Tm = 302.8          # Melting point (K)

# Domain parameters
T0 = 305 #301.3          # Starting temperature (K)
T_H = 311           # Hot wall temperature (K)
T_C = 305 #301.3         # Cold wall temperature (K)
epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
umax = np.sqrt(g * beta * (T_H - T0) * L)           # Maximal velocity
# print(umax)

# Dimensionless numbers
Re = umax * H / nu                                  # Reynolds number
Ra = beta * (T_H - T0) * g * H**3 / (nu * alpha)    # Rayleigh number
print('Ra', Ra)
Pr = nu / alpha                                     # Prandtl number
Ma = 0.1                                            # Mach number

# u_max approximation
c1 = 8.05
c2 = 1.38
c3 = 0.487
c4 = 0.0252
a = 0.922
Re_L = (2*a)**2

def f(x):
    val = (1 + x**4)**(-1/4)

    return val

def g(y):
    val = y * (1 + y**4)**(-1/4)

    return val

def equations(p):
    Nu, Re = p
    return ((Nu - 1) * Ra * Pr**(-2) - c1 * abs(Re)**2 / sqrt(Re_L/abs(Re)) * (1 + sqrt(Re_L/abs(Re))**4)**(-1/4) - c2 * abs(Re)**3, \
            Nu - 1 - c3 * abs(Re)**(1/2) * Pr**(1/2) * f(2*a*Nu/sqrt(Re_L) * sqrt(Re_L/abs(Re)) * (1 + sqrt(Re_L/abs(Re))**4)**(-1/4)) - c4 * Pr * abs(Re) * f(2*a*Nu/sqrt(Re_L) * sqrt(Re_L/abs(Re)) * (1 + sqrt(Re_L/abs(Re))**4)**(-1/4)))

x, y =  fsolve(equations, (10, 5000))

print(x, y)4
