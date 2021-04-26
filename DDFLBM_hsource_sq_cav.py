import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streaming as str
import sys
import time
from matplotlib import cm
from numba import njit, jit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
folder_nr = 'test2'

# Define constants
# # Physical parameters water
# L = 0.1             # Length of cavity (m)
# H = L               # Height of cavity (m)
# g = 9.81            # Gravitational acceleration (m/s^2)
# Time = 400         # (s)
# nu = 1e-6           # Kinematic viscosity (m^2/s)
# alpha = 1.44e-7     # Thermal diffusivity (m^2/s)
# lbda = 0.6          # Thermal conductivity (W/m K)
# rho0 = 1e3          # Density (kg/m^3)
# beta = 210e-6       # Thermal expansion (1/K)
# Lat = 334e3         # Latent heat (J/kg)
# c_p = 4.2e3         # Specific heat (J/(kgK))
# Tm = 273.15         # Melting point (K)
#
# # Domain parameters
# T0 = 275          # Starting temperature (K)
# T_H = 285         # Hot wall temperature (K)
# T_C = 275         # Hot wall temperature (K)
# epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
# umax = np.sqrt(g * beta * (T_H - T0) * L)           # Maximal velocity

# Physical parameters gallium
Time = 1080         # (s)
L = 0.0889             # Length of cavity (m)
H = 0.714*L      # Height of cavity (m)
g = 9.81            # Gravitational acceleration (m/s^2)
rho0 = 6.093e3      # Density (kg/m^3)
lbda = 33           # Thermal conductivity (W/m K)
mu = 1.81e-3        # Dynamic viscosity (Ns/m^2)
nu = mu / rho0      # Kinematic viscosity (m^2/s)
beta = 1.2e-4       # Thermal expansion (1/K)
Lat = 8.016e5       # Latent heat (J/kg)
c_p = 381           # Specific heat (J/(kgK))
alpha = lbda / (rho0 * c_p)     # Thermal diffusivity (m^2/s)
Tm = 290 #302.8          # Melting point (K)

# Domain parameters
T0 = 306 #301.3          # Starting temperature (K)
T_H = 307           # Hot wall temperature (K)
T_C = 305 #301.3         # Hot wall temperature (K)
epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
umax = np.sqrt(g * beta * (T_H - T0) * H)           # Maximal velocity

# Dimensionless numbers
Re = umax * H / nu                                  # Reynolds number
Ra = beta * (T_H - T0) * g * H**3 / (nu * alpha)    # Rayleigh number
print('Ra', Ra)
Pr = nu / alpha                                      # Prandtl number
print('Pr', Pr)

# Choose simulation parameters
Lambda = 1/4        # Magic parameter
tau_plus = 0.501     # Even relaxation time
rho0_sim = 1        # Starting simulation density
Nx = 20             # Nodes in y-direction
Ny = Nx #np.int(0.714*Nx)

dx_sim = 1          # simulation length
dt_sim = 1          # simulation time
c_s = (1 / np.sqrt(3)) * (dx_sim / dt_sim)              # Simulation speed of sound
nu_sim = c_s**2 * (tau_plus - 1 / 2)                    # Simulation viscosity
print('nu_sim', nu_sim)

# Determine dependent parameters
umax_sim = Re * nu_sim / Ny                             # Maximal simulation density
print('umax_sim', umax_sim)
tau_minus = dt_sim * (Lambda / (tau_plus / dt_sim - 1/2) + 1/2)
tau_g_plus = nu_sim / (c_s**2 * Pr) + 1/2
tau_g_minus = tau_g_plus / (2*tau_g_plus - 1)
print('tau_g_plus', tau_g_plus)
print('tau_g_minus', tau_g_minus)
alpha_sim = (tau_g_plus - 1/2) * c_s**2
print('alpha_sim', alpha_sim)

# if alpha_sim > 1/6:
#     print(f"alpha too large ({alpha_sim}), unstable temperature")

# Calculate conversion parameters
dx = L / Nx                                             # Distance
dt = (c_s ** 2) * (tau_plus - 1 / 2) * ((dx ** 2) / nu)       # Time
print(dx, dt)
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
# Ny = np.int(H/dx)                                 # Number of lattice nodes in x-direction
Nt = np.int(Time / dt)                  # Number of time steps
print('Nt', Nt)

# Forces
g_sim = g / Cg * np.array([0, -1])   # Simulation acceleration vector

# Initial conditions
dim = (Nx, Ny)
ux = np.zeros(dim)                 # Simulation velocity in x direction
uy = np.zeros(dim)                 # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones(dim)  # Simulation density
T_dim = np.zeros(dim)     # Dimensionless simulation temperature
f_l_ts = np.ones(dim)                # Liquid fraction
h = (c_p * T0) * np.ones(dim)        # Enthalpy
c_app = c_p * np.ones(dim)

# Temperature BCS
T_dim_H = np.ones(Ny) * beta * (T_H - T0)
T_dim_C = np.ones(Ny) * beta * (T_C - T0)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)

def streaming(Nx, Ny, f_plus, f_minus, f_star):
    f_plus, f_minus = str.fluid(Nx, Ny, f_plus, f_minus, f_star)
    f_plus, f_minus = str.left_wall(Ny, f_plus, f_minus, f_star)
    f_plus, f_minus = str.right_wall(Nx, Ny, f_plus, f_minus, f_star)
    f_plus, f_minus = str.lower_wall(Nx, f_plus, f_minus, f_star)
    f_plus, f_minus = str.upper_wall(Nx, Ny, f_plus, f_minus, f_star)
    f_plus, f_minus = str.lower_left_corner(f_plus, f_minus, f_star)
    f_plus, f_minus = str.lower_right_corner(Nx, f_plus, f_minus, f_star)
    f_plus, f_minus = str.upper_left_corner(Ny, f_plus, f_minus, f_star)
    f_plus, f_minus = str.upper_right_corner(Nx, Ny, f_plus, f_minus, f_star)

    return f_plus, f_minus

def streaming_temp(Nx, Ny, g_plus, g_minus, g_star, w_i, T_H, T_C):
    g_plus, g_minus = str.fluid(Nx, Ny, g_plus, g_minus, g_star)
    g_plus, g_minus = str.left_wall_temp(Ny, g_plus, g_minus, g_star, w_i, T_H)
    g_plus, g_minus = str.right_wall_temp(Nx, Ny, g_plus, g_minus, g_star, w_i, T_C)
    g_plus, g_minus = str.lower_wall(Nx, g_plus, g_minus, g_star)
    g_plus, g_minus = str.upper_wall(Nx, Ny, g_plus, g_minus, g_star)
    g_plus, g_minus = str.lower_left_corner_temp(g_plus, g_minus, g_star, w_i, T_H)
    g_plus, g_minus = str.lower_right_corner_temp(Nx, g_plus, g_minus, g_star, w_i, T_C)
    g_plus, g_minus = str.upper_left_corner_temp(Ny, g_plus, g_minus, g_star, w_i, T_H)
    g_plus, g_minus = str.upper_right_corner_temp(Nx, Ny, g_plus, g_minus, g_star, w_i, T_C)

    return g_plus, g_minus

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
def g_equilibrium(w_i, T, ux, uy, c_i, q, c_s):
    g_eq_plus = np.zeros((Nx, Ny, q))                                   # Initialize even and odd parts of f_eq
    g_eq_minus = np.zeros((Nx, Ny, q))

    u_dot_u = ux**2 + uy**2                                             # Inner product of u with itself

    for i in range(q):                                                  # Loop over all directions of Q
        if i == 0:                                                      # If-statement for symmetry arguments
            u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]                   # Inner product of u with c_i
            g_eq_plus[:, :, i] = w_i[i] * T * (1 + (u_dot_c[:, :] / c_s**2) + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))
        elif i in [1, 2, 5, 6]:
            u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]
            g_eq_plus[:, :, i] = w_i[i] * T * (1 + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))      # Even part of f_eq
            g_eq_minus[:, :, i] = w_i[i] * T * (u_dot_c[:, :] / c_s**2)                                               # Odd part of f_eq
        else:
            g_eq_plus[:, :, i] = g_eq_plus[:, :, c_opp[i]]
            g_eq_minus[:, :, i] = -g_eq_minus[:, :, c_opp[i]]

    return g_eq_plus, g_eq_minus

@njit
def force_source(ux, uy, F):
    Fi = np.zeros((Nx, Ny, q))                                              # Initialize forcing and source terms
    Si = np.zeros((Nx, Ny, q))

    u_dot_F = ux * F[:, :, 0] + uy * F[:, :, 1]                             # Inner product of u with F

    for i in range(q):
        u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]                           # Inner product of u with c_i

        Fi[:, :, i] = F[:, :, 0] * c_i[i, 0] + F[:, :, 1] * c_i[i, 1]       # Inner product of F with c_i
        Si[:, :, i] = (1 - 1/(2*tau_plus)) * w_i[i] * (u_dot_c[:, :] * Fi[:, :, i] / c_s**4 - u_dot_F / c_s**2) + (1 - 1/(2*tau_minus)) * w_i[i] * Fi[:, :, i] / c_s**2   # Source term

    return Si

f_plus, f_minus = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)          # Initialize distributions
g_plus, g_minus = g_equilibrium(w_i, T_dim, ux, uy, c_i, q, c_s)

start = time.time()

X_th = []
X_sim = []
t_phys = []

Nt = 10
for t in range(Nt):
    ### Forcing
    F_buoy = - (T_dim[:, :, None] - np.mean(T_dim)) * g_sim          # Calculate buoyancy force

    ### Moment update
    rho_sim = np.sum(f_plus, axis=2)                                        # Calculate density (even parts due to symmetry)

    B = (1 - f_l_ts) * (tau_plus - 1/2) / (f_l_ts + tau_plus - 1/2)               # Viscosity-dependent solid fraction
    ux = 0 * (np.sum(f_minus[:, :] * c_i[:, 0], axis=2) / rho_sim + (1 - B[:, :]) / 2 * F_buoy[:, :, 0])    # Calculate x velocity (odd parts due to symmetry)
    uy = 0 * (np.sum(f_minus[:, :] * c_i[:, 1], axis=2) / rho_sim + (1 - B[:, :]) / 2 * F_buoy[:, :, 1])    # Calculate y velocity (odd parts due to symmetry)

    ux[np.round(B) == 1] = 0                                                # Force velocity in solid to zero
    uy[B == 1] = 0

    if t % 1 == 0:
        easy_view(t, T_dim / beta + T0)
    # easy_view(t, g_plus[:, :, 1])
    # easy_view(t, g_minus[:, :, 1])

    ### Temperature
    g_eq_plus, g_eq_minus = g_equilibrium(w_i, T_dim, ux, uy, c_i, q, c_s)                                # Calculate new equilibrium distribution
        #
        # easy_view(t, g_eq_plus[:, :, 1])
        # easy_view(t, g_eq_minus[:, :, 1])

    g_star = g_plus + g_minus - (g_plus - g_eq_plus) / tau_g_plus - (g_minus - g_eq_minus) / tau_g_minus
    g_plus, g_minus = streaming_temp(Nx, Ny, g_plus, g_minus, g_star, w_i, T_dim_H, T_dim_C)

    T_dim = np.sum(g_plus, axis=2)

    # T_dim[0, 0] = T_dim[0, 1]
    # T_dim[0, -1] = T_dim[0, -2]
    # T_dim[-1, 0] = T_dim[-1, 1]
    # T_dim[-1, -1] = T_dim[-1, -2]

    ### Equilibrium
    f_eq_plus, f_eq_minus = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)                                # Calculate new equilibrium distribution

    ### Source
    Si = force_source(ux, uy, F_buoy)                      # Calculate source term
    Bi = np.repeat(B[:, :, np.newaxis], q, axis=2)                          # Repeat B in all directions to make next calc possible

    ### Collision
    f_star = f_plus * (1 - (1-Bi) / tau_plus) + f_minus * (1 - 2*Bi - (1-Bi) / tau_minus) + f_eq_plus * (1-Bi) / tau_plus + f_eq_minus * (1-Bi) / tau_minus + Si * (1-Bi)

    ### Streaming
    f_plus, f_minus = streaming(Nx, Ny, f_plus, f_minus, f_star)

    # ### Plots
    if (t % 5000 == 0):
        T = T_dim / beta + T0
        x = np.linspace(L/Nx/2, L-L/Nx/2, len(T[5, :]))

        # ### Moment update
        # rho_sim = np.sum(f_plus, axis=2)                                        # Calculate density (even parts due to symmetry)
        #
        # B = (1 - f_l_ts) * (tau_plus - 1/2) / (f_l_ts + tau_plus - 1/2)               # Viscosity-dependent solid fraction
        # ux = 1 * (np.sum(f_minus[:, :] * c_i[:, 0], axis=2) / rho_sim)    # Calculate x velocity (odd parts due to symmetry)
        # uy = 1 * (np.sum(f_minus[:, :] * c_i[:, 1], axis=2) / rho_sim)    # Calculate y velocity (odd parts due to symmetry)
        #
        # ux[np.round(B) == 1] = 0                                                # Force velocity in solid to zero
        # uy[B == 1] = 0

        # easy_view(t, T)
        # easy_view(t, f_l_ts)
        # easy_view(t, uy)

        # # Liquid fraction
        # plt.figure()
        # plt.imshow(f_l_ts.T, cmap=cm.autumn, origin='lower', aspect=1.0)
        # plt.xlabel('$x$ (# lattice nodes)')
        # plt.ylabel('$y$ (# lattice nodes)')
        # plt.title(f'Gallium \n $f_l$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.colorbar()
        # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}_test.png")

        # Velocities
        plt.figure()
        plt.clf()
        plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Gallium \n $u_y$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}_test.png")

        # plt.figure()
        # plt.clf()
        # plt.imshow(ux.T, cmap=cm.Blues, origin='lower')
        # plt.xlabel('$x$ (# lattice nodes)')
        # plt.ylabel('$y$ (# lattice nodes)')
        # plt.title(f'Gallium \n $u_x$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.colorbar()
        # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_ux_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")

        ## Temperature heatmap
        plt.figure()
        plt.clf()
        plt.imshow(np.flip(T[1:-1, 1:-1].T, axis=0), cmap=cm.Blues)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Gallium \n $T$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_T_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")

        # ## Temperature lineplot
        # plt.figure()
        # plt.plot(T[5, :], x)
        # plt.xlabel('$x$ (m)')
        # plt.ylabel('$T$ (K)')
        # plt.title(f'Gallium \n $T$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/lineplot_T_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")


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
        plt.title(f'Gallium \n $u$ in pipe with left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.legend('Velocity vector')
        plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/arrowplot_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")

        plt.close('all')


# # Make arrays from lists
# t_phys = np.array(t_phys)
# X_th = np.array(X_th)
# X_sim = np.array(X_sim)
# plt.figure()
# plt.plot(X_th, t_phys)
# plt.plot(X_sim, t_phys)
# plt.xlabel('$x$ (m)')
# plt.ylabel('$t$ (s)')
# plt.title(f'Gallium \n Position of melting front, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/x_pos_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_adv_form.png")
#
# # Liquid fraction
# plt.figure()
# plt.imshow(f_l_ts.T, cmap=cm.autumn, origin='lower', aspect=1.0)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'Gallium \n $f_l$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_adv_form.png")

stop = time.time()
print(stop-start)
