import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from matplotlib import cm
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
folder_nr = 'water_Hsource'

# Define constants
# Physical parameters water
L = 0.1             # Length of cavity (m)
H = L               # Height of cavity (m)
g = 9.81            # Gravitational acceleration (m/s^2)
Time = 2000          # (s)
nu = 1e-6           # Kinematic viscosity (m^2/s)
alpha = 1.44e-7     # Thermal diffusivity (m^2/s)
rho0 = 1e3          # Density (kg/m^3)
beta = 210e-6       # Thermal expansion (1/K)
Lat = 334e3         # Latent heat (J/kg)
c_p = 4.2e3         # Specific heat (J/(kgK))
Tm = 273.15         # Melting point (K)

# Domain parameters
T0 = 272          # Starting temperature (K)
T_H = 280         # Hot wall temperature (K)
T_C = 272         # Hot wall temperature (K)
epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
umax = np.sqrt(g * beta * (T_H - T0) * L)           # Maximal velocity

# # Physical parameters gallium
# Time = 200          # (s)
# L = 0.1             # Length of cavity (m)
# H = 0.714*L         # Height of cavity (m)
# g = 9.81            # Gravitational acceleration (m/s^2)
# rho0 = 1e3          # Density (kg/m^3)
# lbda = 33           # Thermal conductivity
# mu = 1.81e-3        # Dynamic viscosity (Ns/m^2)
# nu = mu/rho0        # Kinematic viscosity (m^2/s)
# beta = 1.2e-4       # Thermal expansion (1/K)
# Lat = 8.016e5       # Latent heat (J/kg)
# c_p = 381           # Specific heat (J/(kgK))
# alpha = lbda / (rho0 * c_p)     # Thermal diffusivity (m^2/s)
# Tm = 302.8          # Melting point (K)
#
# # Domain parameters
# T0 = 301.3          # Starting temperature (K)
# T_H = 311           # Hot wall temperature (K)
# T_C = 301.3         # Hot wall temperature (K)
# epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
# umax = np.sqrt(g * beta * (T_H - T0) * L)           # Maximal velocity

# Dimensionless numbers
Re = umax * H / nu                                  # Reynolds number
Ra = beta * (T_H - T0) * g * H**3 / (nu * alpha)    # Rayleigh number
print('Ra', Ra)
Pr = nu / alpha                                     # Prandtl number
Ma = 0.1                                            # Mach number

# Choose simulation parameters
Lambda = 1/4        # Magic parameter
tau_plus = 0.51     # Even relaxation time
rho0_sim = 1        # Starting simulation density
Ny = 20             # Nodes in y-direction

dx_sim = 1          # simulation length
dt_sim = 1          # simulation time
c_s = (1 / np.sqrt(3)) * (dx_sim / dt_sim)              # Simulation speed of sound
nu_sim = c_s**2 * (tau_plus - 1 / 2)                    # Simulation viscosity
print('nu_sim', nu_sim)

# Determine dependent parameters
umax_sim = Re * nu_sim / Ny                             # Maximal simulation density
print('umax_sim', umax_sim)
tau_minus = dt_sim * (Lambda / (tau_plus / dt_sim - 1/2) + 1/2)
alpha_sim = nu_sim / Pr

if alpha_sim > 1/6:
    print(f"alpha too large ({alpha_sim}), unstable temperature")

# Calculate conversion parameters
dx = H / Ny                                             # Distance
dt = (c_s ** 2) * (tau_plus - 1 / 2) * ((dx ** 2) / nu)       # Time
Cu = dx / dt                                            # Velocity
Cg = dx / dt**2                                         # Acceleration
Crho = rho0 / rho0_sim                                  # Density
CF = Crho * Cg                                          # Force
Ch = dx**2 / dt**2                                      # Specific enthalpy

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
q = 9                               # number of directions

# Grid and time steps
Nx = Ny                                 # Number of lattice nodes in x-direction
Nt = np.int(Time / dt)                  # Number of time steps
print('Nt', Nt)

# Forces
g_sim = g / Cg * np.array([0, -1])      # Simulation acceleration vector

# Initial conditions
dim = (Nx, Ny)
ux = np.zeros(dim)                 # Simulation velocity in x direction
uy = np.zeros(dim)                 # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones(dim)  # Simulation density
T_dim = np.zeros((Nx+2, Ny+2))     # Dimensionless simulation temperature
f_l = np.zeros(dim)                # Liquid fraction
f_l_t_1 = np.zeros(dim)            # Liquid fraction one time step before

# Temperature BCS
T_dim_H = np.ones(Ny+2) * beta * (T_H - T0)
T_dim_C = np.ones(Ny+2) * beta * (T_C - T0)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)

@njit
def fluid(Nx, Ny, f_i, f_star):
    for i in range(1, Nx-1):                # Streaming in all nodes expect ones nearest to the wall
        f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
        f_i[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
        f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
        f_i[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
        f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
        f_i[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 5]
        f_i[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
        f_i[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
        f_i[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

    return f_i

@njit
def left_wall(Ny, f_i, f_star):             # Streaming in nodes touching left wall (half-way bounce-back)
    i = 0

    f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f_i[i, 1:Ny-1, 1] = f_star[i, 1:Ny-1, 3]        # Bounce
    f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f_i[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
    f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f_i[i, 1:Ny-1, 5] = f_star[i, 1:Ny-1, 7]        # Bounce
    f_i[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
    f_i[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
    f_i[i, 1:Ny-1, 8] = f_star[i, 1:Ny-1, 6]        # Bounce

    return f_i

@njit
def right_wall(Nx, Ny, f_i, f_star):        # Streaming in nodes touching right wall (half-way bounce-back)
    i = Nx - 1

    f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f_i[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
    f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f_i[i, 1:Ny-1, 3] = f_star[i, 1:Ny-1, 1]        # Bounce
    f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f_i[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 5]
    f_i[i, 1:Ny-1, 6] = f_star[i, 1:Ny-1, 8]        # Bounce
    f_i[i, 1:Ny-1, 7] = f_star[i, 1:Ny-1, 5]        # Bounce
    f_i[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

    return f_i

@njit
def lower_wall(Nx, f_i, f_star):            # Streaming in nodes touching lower wall (half-way bounce-back)
    j = 0

    f_i[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f_i[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f_i[1:Nx-1, j, 2] = f_star[1:Nx-1, j, 4]        # Bounce
    f_i[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f_i[1:Nx-1, j, 4] = f_star[1:Nx-1, j+1, 4]
    f_i[1:Nx-1, j, 5] = f_star[1:Nx-1, j, 7]        # Bounce
    f_i[1:Nx-1, j, 6] = f_star[1:Nx-1, j, 8]        # Bounce
    f_i[1:Nx-1, j, 7] = f_star[2:Nx, j+1, 7]
    f_i[1:Nx-1, j, 8] = f_star[0:Nx-2, j+1, 8]

    return f_i

@njit
def upper_wall(Nx, Ny, f_i, f_star):        # Streaming in nodes touching upper wall (half-way bounce-back)
    j = Ny - 1

    f_i[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f_i[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f_i[1:Nx-1, j, 2] = f_star[1:Nx-1, j-1, 2]
    f_i[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f_i[1:Nx-1, j, 4] = f_star[1:Nx-1, j, 2]        # Bounce
    f_i[1:Nx-1, j, 5] = f_star[0:Nx-2, j-1, 5]
    f_i[1:Nx-1, j, 6] = f_star[2:Nx, j-1, 6]
    f_i[1:Nx-1, j, 7] = f_star[1:Nx-1, j, 5]        # Bounce
    f_i[1:Nx-1, j, 8] = f_star[1:Nx-1, j, 6]        # Bounce

    return f_i

@njit
def lower_left_corner(f_i, f_star):         # Streaming in node lower left corner (half-way bounce-back)
    i = 0
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j+1, 4]
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i+1, j+1, 7]
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@njit
def lower_right_corner(Nx, f_i, f_star):    # Streaming in node lower right corner (half-way bounce-back)
    i = Nx - 1
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j+1, 4]
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i-1, j+1, 8]

    return f_i

@njit
def upper_left_corner(f_i, f_star):         # Streaming in node upper left corner (half-way bounce-back)
    i = 0
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i+1, j-1, 6]
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@njit
def upper_right_corner(f_i, f_star):        # Streaming in node upper right corner (half-way bounce-back)
    i = Nx - 1
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    f_i[i, j, 5] = f_star[i-1, j-1, 5]
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

def streaming(Nx, Ny, f_i, f_star):         # Function to access all streaming functions
    f_i = fluid(Nx, Ny, f_i, f_star)
    f_i = left_wall(Ny, f_i, f_star)
    f_i = right_wall(Nx, Ny, f_i, f_star)
    f_i = lower_wall(Nx, f_i, f_star)
    f_i = upper_wall(Nx, Ny, f_i, f_star)
    f_i = lower_left_corner(f_i, f_star)
    f_i = lower_right_corner(Nx, f_i, f_star)
    f_i = upper_left_corner(f_i, f_star)
    f_i = upper_right_corner(f_i, f_star)

    return f_i

@njit
def f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s):
    f_eq_plus = np.zeros((Nx, Ny, q))                                   # Initialize even and odd parts of f_eq
    f_eq_minus = np.zeros((Nx, Ny, q))

    u_dot_u = ux**2 + uy**2                                             # Inner product of u with itself

    for i in range(q):                                                  # Loop over all directions of Q
        if i == 0:                                                      # If-statement for symmetry arguments
            u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]                   # Inner product of u with c_i
            f_eq_plus[:, :, i] = w_i[i] * rho * (1 + (u_dot_c[:, :] / c_s**2) + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))
        elif i in [1, 2, 5, 6]:
            u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]
            f_eq_plus[:, :, i] = w_i[i] * rho * (1 + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))      # Even part of f_eq
            f_eq_minus[:, :, i] = w_i[i] * rho * (u_dot_c[:, :] / c_s**2)                                               # Odd part of f_eq
        else:
            f_eq_plus[:, :, i] = f_eq_plus[:, :, c_opp[i]]
            f_eq_minus[:, :, i] = -f_eq_minus[:, :, c_opp[i]]

    return f_eq_plus, f_eq_minus

@njit
def force_source(w_i, c_i, c_s, tau_plus, F):
    Fi = np.zeros((Nx, Ny, q))                                              # Initialize forcing and source terms
    Si = np.zeros((Nx, Ny, q))

    u_dot_F = ux * F[:, :, 0] + uy * F[:, :, 1]                             # Inner product of u with F

    for i in range(q):
        u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]                           # Inner product of u with c_i

        Fi[:, :, i] = F[:, :, 0] * c_i[i, 0] + F[:, :, 1] * c_i[i, 1]       # Inner product of F with c_i
        Si[:, :, i] = (tau_plus - 1/2) * w_i[i] * (u_dot_c[:, :] * Fi[:, :, i] / c_s**2 - u_dot_F) + (tau_minus - 1/2) * w_i[i] * Fi[:, :, i]   # Source term

    return Si

@njit
def decompose_f_i(q, f_plus, f_minus, f_i):
    for i in range(q):                  # Decompose f_i for every direction
        if i == 0:                      # If-statement for symmetry arguments
            f_plus[:, :, i] = f_i[:, :, i]
            f_minus[:, :, i] = 0
        elif i in [1, 2, 5, 6]:
            f_plus[:, :, i] = (f_i[:, :, i] + f_i[:, :, c_opp[i]]) / 2
            f_minus[:, :, i] = f_i[:, :, i] - f_plus[:, :, i]
        else:
            f_plus[:, :, i] = f_plus[:, :, c_opp[i]]
            f_minus[:, :, i] = -f_minus[:, :, c_opp[i]]

    return f_plus, f_minus


def temperature(T, alpha, Lat, c_p, beta, ux, uy, t, T_dim_C, T_dim_H, f_l_t_1, f_l_t_2):
    T_new = np.zeros((Nx+2, Ny+2))
    f_l = f_l_t_1.copy()

    Tm_sim = beta * (Tm - T0)
    epsilon_sim = epsilon * beta

    def energy_eq(i, j, T, ux, uy, f_l_old_iter, f_l_old_time):
        T_cell = (1 - 6 * alpha) * T[i, j] + (2 * alpha - ux[i-1, j-1]) * T[i+1, j] \
                      + (2 * alpha + ux[i-1, j-1]) * T[i-1, j] + (2 * alpha - uy[i-1, j-1]) * T[i, j+1] \
                      + (2 * alpha + uy[i-1, j-1]) * T[i, j-1] + (ux[i-1, j-1] / 4 + uy[i-1, j-1] / 4 - alpha / 2) * T[i+1, j+1] \
                      + (-ux[i-1, j-1] / 4 + uy[i-1, j-1] / 4 - alpha / 2) * T[i-1, j+1] + (ux[i-1, j-1] / 4 - uy[i-1, j-1] / 4 - alpha / 2) * T[i+1, j-1] \
                      + (-ux[i-1, j-1] / 4 - uy[i-1, j-1] / 4 - alpha / 2) * T[i-1, j-1] - beta * Lat / c_p * (f_l_old_iter - f_l_old_time)

        return T_cell

    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            f_l_old_iter = f_l_t_1[i-1, j-1].copy()
            f_l_old_time = f_l_t_2[i-1, j-1].copy()

            T_new[i, j] = energy_eq(i, j, T, ux, uy, f_l_old_iter, f_l_old_time)

            if T_new[i, j] < (Tm_sim-epsilon_sim):
                f_l[i-1, j-1] = 0
            elif T_new[i, j] > (Tm_sim+epsilon_sim):
                f_l[i-1, j-1] = 1
            else:
                x = 0
                while True:
                    x += 1
                    T_new[i, j] = energy_eq(i, j, T, ux, uy, f_l_old_iter, f_l_old_time)

                    # T_prime = Tm - epsilon + 2 * epsilon * f_l_t_1[i-1, j-1]
                    T_prime = Tm - epsilon + 2 * epsilon * f_l_old_iter
                    T_new_phys = T_new[i, j] / beta + T0

                    f_l[i-1, j-1] = f_l_old_iter + 0.1 * c_p / Lat * (T_new_phys - T_prime)

                    if np.abs(f_l[i-1, j-1] - f_l_old_iter) < 1e-5:
                        f_l[i-1, j-1] = min(max(f_l[i-1, j-1], 0), 1)
                        break

                    f_l_old_time = f_l_t_1[i-1, j-1].copy()
                    f_l_old_iter = f_l[i-1, j-1].copy()

    # Ghost nodes
    T_new[1:-1, 0] = 21/23 * T_new[1:-1, 1] + 3/23 * T_new[1:-1, 2] - 1/23 * T_new[1:-1, 3]         # Neumann extrapolation on lower boundary
    T_new[1:-1, -1] = 21/23 * T_new[1:-1, -2] + 3/23 * T_new[1:-1, -3] - 1/23 * T_new[1:-1, -4]     # Neumann extrapolation on upper boundary
    # T_new[-1, :] = 21/23 * T_new[-2, :] + 3/23 * T_new[-3, :] - 1/23 * T_new[-4, :]               # Neumann extrapolation on right boundary
    T_new[0, :] = 16/15 * T_dim_H - 3 * T_new[1, :] + T_new[2, :] - 1/5 * T_new[3, :]               # Dirichlet extrapolation on left boundary
    T_new[-1, :] = 16/15 * T_dim_C - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]           # Dirichlet extrapolation on right boundary

    # if (t % 2500 == 0):
    # if t > 6350 and t < 6360:
    #     easy_view(t, T_new / beta + T0)
    #     easy_view(t, f_l)

    return T_new, f_l, f_l_t_1


# Buoyancy force
F_buoy = - T_dim[1:-1, 1:-1, None] * g_sim

# Initialize equilibrium function
f_plus, f_minus = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)
f_i = f_plus + f_minus

start = time.time()                 # Start timer

f_star = np.zeros((Nx, Ny, q))      # Initialize f_star

for t in range(Nt):
    B = (1 - f_l) * (tau_plus - 1/2) / (f_l + tau_plus - 1/2)               # Viscosity-dependent solid fraction

    # Buoyancy force
    T_dim_phys = T_dim[1:-1, 1:-1]                                          # Select only physical domain w/out ghost nodes

    if np.any(f_l[f_l == 1]):                                               # Check if at least one node is fully liquid
        T_dim_avg_l = np.mean(T_dim_phys[f_l == 1])                              # Take average of all liquid nodes
        F_buoy = - (T_dim[1:-1, 1:-1, None] - T_dim_avg_l) * g_sim                # Calculate buoyancy force
    else:
        F_buoy = - T_dim[1:-1, 1:-1, None] * g_sim                                # Calculate buoyancy force

    # Calculate macroscopic quantities
    rho_sim = np.sum(f_plus, axis=2)                                        # Calculate density (even parts due to symmetry)

    ux = 0 * (np.sum(f_minus[:, :] * c_i[:, 0], axis=2) / rho_sim + (1 - B[:, :]) / 2 * F_buoy[:, :, 0])  # Calculate x velocity (odd parts due to symmetry)
    uy = 0 * (np.sum(f_minus[:, :] * c_i[:, 1], axis=2) / rho_sim + (1 - B[:, :]) / 2 * F_buoy[:, :, 1])  # Calculate y velocity (odd parts due to symmetry)

    ux[np.round(B) == 1] = 0      # Force velocity in solid to zero
    uy[B == 1] = 0

    # Calculate temperature and liquid fraction
    T_dim, f_l, f_l_t_1 = temperature(T_dim, alpha_sim, Lat, c_p, beta, ux, uy, t, T_dim_C, T_dim_H, f_l, f_l_t_1)

    # Calculate new equilibrium distribution
    f_eq_plus, f_eq_minus = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)

    # Collision step
    Si = force_source(w_i, c_i, c_s, tau_plus, F_buoy)          # Calculate source term
    Bi = np.repeat(B[:, :, np.newaxis], q, axis=2)              # Repeat B in all directions to make next calc possible
    # f_star = f_plus + f_minus + (1-Bi) * (-1 / tau_plus * (f_plus - f_eq_plus) - 1 / tau_minus * (f_minus - f_eq_minus)) + Bi * (f_plus - f_minus - f_plus + f_minus) + (1-Bi) * Si
    # if t > 1:
    #     easy_view(t, T)
    #     easy_view(t, f_plus[:, :, 1])
    #     easy_view(t, Bi[:, :, 1])
    #     easy_view(t, f_minus[:, :, 1])
    #     easy_view(t, f_eq_plus[:, :, 1])
    #     easy_view(t, f_eq_minus[:, :, 1])

    f_star = f_plus * (1 - (1-Bi) / tau_plus) + f_minus * (1 - 2*Bi - (1-Bi) / tau_minus) + f_eq_plus * (1-Bi) / tau_plus + f_eq_minus * (1-Bi) / tau_minus + Si * (1-Bi)

    # Streaming step
    f_i = streaming(Nx, Ny, f_i, f_star)
    f_plus, f_minus = decompose_f_i(q, f_plus, f_minus, f_i)

    if (t % 10000 == 1):
        T = T_dim / beta + T0

        # Liquid fraction
        plt.figure()
        plt.imshow(f_l.T, cmap=cm.Blues, origin='lower', aspect=1.0)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'$f_l$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}.png")

        # Velocities
        plt.figure()
        plt.clf()
        plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'$u_y$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}.png")

        plt.figure()
        plt.clf()
        plt.imshow(ux.T, cmap=cm.Blues, origin='lower')
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'$u_x$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_ux_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}.png")

        ## Temperature heatmap
        plt.figure()
        plt.clf()
        plt.imshow(np.flip(T[1:-1, 1:-1].T, axis=0), cmap=cm.Blues)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'$T$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_T_t={np.round(t/Nt*Time, decimals=3)}_test.png")

        # plt.figure()
        # plt.clf()
        # plt.imshow(np.flip(rho_sim, axis=1).T, cmap=cm.Blues)
        # plt.xlabel('$x$ (# lattice nodes)')
        # plt.ylabel('$y$ (# lattice nodes)')
        # plt.title(f'$\\rho$ in cavity with left wall at $T={T_H}K$')
        # plt.colorbar()
        # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_rho_time{Time}_t={np.round(t/Nt*Time, decimals=3)}.png")

        # Vector plot
        plt.figure()
        plt.quiver(Cu*ux.T, Cu*uy.T)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'$u$ in pipe with left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.legend('Velocity vector')
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/arrowplot_time{Time}_test.png")

        plt.close('all')

        # easy_view('uy', uy)
        # easy_view('ux', ux)
        # easy_view('B', B)

stop = time.time()
print(stop-start)

r = np.linspace(-Ny/2, Ny/2, num=Ny)
r[0] = r[0] + (r[1] - r[0]) / 2
r[-1] = r[-1] + (r[-2] - r[-1]) / 2

r_phys = r*dx
R = r_phys[-1]

umax_sim = np.amax(ux[np.int(np.rint(Nx / 2)), 1:Ny])
u_th = umax_sim * (1 - r_phys ** 2 / R ** 2)

T = T_dim / beta + T0

# ## Streamplot
# plt.figure(figsize=(7,9))
# x = np.linspace(dx, L-dx, len(ux))
# y = np.linspace(dx, H-dx, len(uy))
# X, Y = np.meshgrid(x, y)
# plt.streamplot(X, Y, ux.T, uy.T)
# plt.show()

## Stream plot
# # plt.figure(np.int(t/200)+1)
# x = np.linspace(dx, L-dx, len(ux))
# y = np.linspace(dx, H-dx, len(uy))
# fig = ff.create_streamline(x, y, ux.T, uy.T)
# # fig.show()
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
# plt.legend('Velocity vector')
# plt.savefig("Figures/hsource_trt-fsm_sq_cav/arrowplot_temp" + str(t-2) + ".png")

# # Vector plot
# plt.figure(np.int(t/200)+2, dpi=300)
# plt.quiver(Cu*ux.T, Cu*uy.T)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u$ in pipe with left wall at $T={T_H}K$')
# # plt.legend('Velocity vector')
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/arrowplot_time{Time}_test.png")
#
# Heatmaps
# plt.figure(1)
# plt.clf()
# plt.imshow(f_l.T, cmap=cm.Blues, origin='lower')
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'Liquid fraction, left wall at $T={T_H}K$, $t={Nt/t}$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_fl_time{Time}_test2.png")
# Heatmaps
# plt.figure(2)
# plt.clf()
# plt.imshow(Cu*ux.T, cmap=cm.Blues, origin='lower')
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u_x$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_ux_time{Time}_test.png")
#
# plt.figure(3)
# plt.clf()
# plt.imshow(np.flip(Cu*uy, axis=1).T, cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u_y$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_uy_time{Time}_test.png")

# plt.figure(5)
# plt.clf()
# plt.imshow(np.flip(rho_sim, axis=1).T, cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$\\rho$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_rho_time{Time}_test2.png")
#
# ## Temperature heatmap
# plt.figure(4)
# plt.clf()
# plt.imshow(np.flip(T.T, axis=0), cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('$T$ in cavity with left wall at $T=298K$ and right wall at $T=288K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_T_time{Time}_test2.png")

# ## Temperature line plot
# plt.figure(0)
# plt.plot(x, T[:, 0])
# plt.xlabel('$x$ (m)')
# plt.ylabel('$T$ (K)')
# plt.title('$T$ in cavity with left wall at $T=298K$ and right wall at $T=288K$. 1D cross section.')
# plt.savefig(f"Figures/hsource_sq_cav/{folder_nr}/lineplot_T_adv_newBCs_time{Time}" + str() + ".png")
