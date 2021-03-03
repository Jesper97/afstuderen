import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import sys
import time
from matplotlib import cm
from numba import jit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Define constants
# Physical parameters
L = 0.0075            # m
H = L            # m
g = 9.81            # m/s^2
Time = 10           # s
nu = 1e-6           # m^2/s
alpha = 1.44e-7     # m^2/s
rho0 = 1e3          # kg/m^3
beta = 210e-6       # 1/K
Lat = 334e3           # Latent heat (J/kg)
c_p = 4.2e3         # Specific heat (J/(kgK))
Tm = 273.15         # Melting point (K)
h_s = c_p * Tm      # Specific enthalpy of solid (J/kg)
h_l = h_s + Lat     # Specific enthalpy of liquid (J/kg)

T0 = 271            # K
T_C = 271           # K
T_H = 280           # K
umax = np.sqrt(g * beta * (T_H - T0) * L)

# Dimensionless numbers
Re = umax * H / nu
Ra = beta * (T_H - T0) * g * H**3 / (nu * alpha)
print('Ra', Ra)
Pr = 7
Ma = 0.1

# Choose simulation parameters
Lambda = 1/4
tau_plus = 0.55
rho0_sim = 1
Ny = 80

dx_sim = 1          # simulation length
dt_sim = 1          # simulation time
c_s = (1 / np.sqrt(3)) * (dx_sim / dt_sim)  # speed of sound
nu_sim = c_s**2 * (tau_plus - 1 / 2)
print('nu_sim', nu_sim)

# Determine dependent parameters
umax_sim = Re * nu_sim / Ny
print('umax_sim', umax_sim)
tau_minus = dt_sim * (Lambda / (tau_plus / dt_sim - 1/2) + 1/2)

# Calculate conversion parameters
dx = H / Ny
dt = c_s ** 2 * (tau_plus - 1 / 2) * dx ** 2 / nu
Cu = dx / dt
Cg = dx / dt**2
Crho = rho0 / rho0_sim
CF = Crho * Cg
Ch = dx**2 / dt**2

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
q = 9                               # number of directions

# Simulation parameters
alpha_sim = nu_sim / Pr
h_s_sim = h_s / Ch
h_l_sim = (h_s + Lat) / Ch
Lat_sim = Lat / Ch

if alpha_sim > 1/6:
    print("alpha too large, unstable temperature")

# Grid and time steps
Nx = Ny                     # lattice nodes in the x-direction
Nt = np.int(Time / dt)         # time steps
print('Nt', Nt)

# Forces
g_sim = g / Cg * np.array([0, -1])

# Initial conditions
ux = np.zeros((Nx, Ny))                 # Simulation velocity in x direction
uy = np.zeros((Nx, Ny))                 # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones((Nx, Ny))  # Simulation density
T_dim = np.zeros((Nx, Ny))              # Dimensionless simulation temperature
f_l = np.zeros((Nx, Ny))                # Liquid fraction

# Temperature BCS
T_BC_H = np.ones(Ny) * beta * (T_H - T0)
T_BC_C = np.ones(Ny) * beta * (T_C - T0)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)

@jit
def fluid(Nx, Ny, f_i, f_star):
    for i in range(1, Nx-1):
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

@jit
def left_wall(Ny, f_i, f_star):
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

@jit
def right_wall(Nx, Ny, f_i, f_star):
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

@jit
def lower_wall(Nx, f_i, f_star):
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

@jit
def upper_wall(Nx, Ny, f_i, f_star):
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

@jit
def lower_left_corner(f_i, f_star):
    i = 0
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j+1, 4]
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    # f_i[i, j, 6] = 0
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i+1, j+1, 7]
    # f_i[i, j, 8] = 0
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@jit
def lower_right_corner(Nx, f_i, f_star):
    i = Nx - 1
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j+1, 4]
    # f_i[i, j, 5] = 0
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    # f_i[i, j, 7] = 0
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i-1, j+1, 8]

    return f_i

@jit
def upper_left_corner(f_i, f_star):
    i = 0
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    # f_i[i, j, 5] = 0
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i+1, j-1, 6]
    # f_i[i, j, 7] = 0
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@jit
def upper_right_corner(f_i, f_star):
    i = Nx - 1
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    f_i[i, j, 5] = f_star[i-1, j-1, 5]
    # f_i[i, j, 6] = 0
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    # f_i[i, j, 8] = 0
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

def streaming(Nx, Ny, f_i, f_star):
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

@jit
def f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s, F):
    f_eq = np.zeros((Nx, Ny, q))
    f_eq_plus = np.zeros((Nx, Ny, q))
    f_eq_minus = np.zeros((Nx, Ny, q))
    u_dot_c = np.zeros((Nx, Ny, q))
    Fi = np.zeros((Nx, Ny, q))
    Si = np.zeros((Nx, Ny, q))

    u_dot_u = ux**2 + uy**2
    u_dot_F = ux * F[:, :, 0] + uy * F[:, :, 1]

    for i in range(q):
        u_dot_c[:, :, i] = ux * c_i[i, 0] + uy * c_i[i, 1]
        f_eq[:, :, i] = w_i[i] * rho * (1 + (u_dot_c[:, :, i] / c_s**2) + (u_dot_c[:, :, i]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))

        Fi[:, :, i] = F[:, :, 0] * c_i[i, 0] + F[:, :, 1] * c_i[i, 1]
        Si[:, :, i] = (tau_plus - 1/2) * w_i[i] * (u_dot_c[:, :, i] * Fi[:, :, i] / c_s**2 - u_dot_F) + (tau_minus - 1/2) * w_i[i] * Fi[:, :, i]

    f_eq_plus[:, :, 0] = f_eq[:, :, 0]
    f_eq_plus[:, :, 1] = (f_eq[:, :, 1] + f_eq[:, :, 3]) / 2
    f_eq_plus[:, :, 2] = (f_eq[:, :, 2] + f_eq[:, :, 4]) / 2
    f_eq_plus[:, :, 3] = f_eq_plus[:, :, 1]
    f_eq_plus[:, :, 4] = f_eq_plus[:, :, 2]
    f_eq_plus[:, :, 5] = (f_eq[:, :, 5] + f_eq[:, :, 7]) / 2
    f_eq_plus[:, :, 6] = (f_eq[:, :, 6] + f_eq[:, :, 8]) / 2
    f_eq_plus[:, :, 7] = f_eq_plus[:, :, 5]
    f_eq_plus[:, :, 8] = f_eq_plus[:, :, 6]

    f_eq_minus[:, :, 0] = 0
    f_eq_minus[:, :, 1] = (f_eq[:, :, 1] - f_eq[:, :, 3]) / 2
    f_eq_minus[:, :, 2] = (f_eq[:, :, 2] - f_eq[:, :, 4]) / 2
    f_eq_minus[:, :, 3] = -f_eq_minus[:, :, 1]
    f_eq_minus[:, :, 4] = -f_eq_minus[:, :, 2]
    f_eq_minus[:, :, 5] = (f_eq[:, :, 5] - f_eq[:, :, 7]) / 2
    f_eq_minus[:, :, 6] = (f_eq[:, :, 6] - f_eq[:, :, 8]) / 2
    f_eq_minus[:, :, 7] = -f_eq_minus[:, :, 5]
    f_eq_minus[:, :, 8] = -f_eq_minus[:, :, 6]

    return f_eq_plus, f_eq_minus, Si

@jit
def decompose_f_i(f_i):
    f_plus = np.zeros((Nx, Ny, q))
    f_minus = np.zeros((Nx, Ny, q))

    f_plus[:, :, 0] = f_i[:, :, 0]
    f_plus[:, :, 1] = (f_i[:, :, 1] + f_i[:, :, 3]) / 2
    f_plus[:, :, 2] = (f_i[:, :, 2] + f_i[:, :, 4]) / 2
    f_plus[:, :, 3] = f_plus[:, :, 1]
    f_plus[:, :, 4] = f_plus[:, :, 2]
    f_plus[:, :, 5] = (f_i[:, :, 5] + f_i[:, :, 7]) / 2
    f_plus[:, :, 6] = (f_i[:, :, 6] + f_i[:, :, 8]) / 2
    f_plus[:, :, 7] = f_plus[:, :, 5]
    f_plus[:, :, 8] = f_plus[:, :, 6]

    f_minus[:, :, 0] = 0
    f_minus[:, :, 1] = (f_i[:, :, 1] - f_i[:, :, 3]) / 2
    f_minus[:, :, 2] = (f_i[:, :, 2] - f_i[:, :, 4]) / 2
    f_minus[:, :, 3] = -f_minus[:, :, 1]
    f_minus[:, :, 4] = -f_minus[:, :, 2]
    f_minus[:, :, 5] = (f_i[:, :, 5] - f_i[:, :, 7]) / 2
    f_minus[:, :, 6] = (f_i[:, :, 6] - f_i[:, :, 8]) / 2
    f_minus[:, :, 7] = -f_minus[:, :, 5]
    f_minus[:, :, 8] = -f_minus[:, :, 6]

    return f_plus, f_minus


def temperature(T, alpha, ux, uy, T_BC_C, T_BC_H):
    T_new = np.zeros((Nx, Ny))

    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            T_new[i, j] = (1 - 6 * alpha) * T[i, j] + (2 * alpha - ux[i, j]) * T[i+1, j] \
                          + (2 * alpha + ux[i, j]) * T[i-1, j] + (2 * alpha - uy[i, j]) * T[i, j+1] \
                          + (2 * alpha + uy[i, j]) * T[i, j-1] + (ux[i, j] / 4 + uy[i, j] / 4 - alpha / 2) * T[i+1, j+1] \
                          + (-ux[i, j] / 4 + uy[i, j] / 4 - alpha / 2) * T[i-1, j+1] + (ux[i, j] / 4 - uy[i, j] / 4 - alpha / 2) * T[i+1, j-1] \
                          + (-ux[i, j] / 4 - uy[i, j] / 4 - alpha / 2) * T[i-1, j-1]

    # T BCs
    T_new[0, 1:Ny-1] = 8/15 * T_BC_C[1:Ny-1] + 2/3 * T_new[1, 1:Ny-1] - 1/5 * T_new[2, 1:Ny-1]
    T_new[-1, 1:Ny-1] = 8/15 * T_BC_H[1:Ny-1] + 2/3 * T_new[-2, 1:Ny-1] - 1/5 * T_new[-3, 1:Ny-1]

    # Adiabatic BCs
    T_new[:, 0] = 3/2 * T_new[:, 1] - 1/2 * T_new[:, 2]
    T_new[:, -1] = 3/2 * T_new[:, -2] - 1/2 * T_new[:, -3]

    return T_new


def enthalpy(T_dim, h_s, h_l, f_l):
    h = c_p * (T_dim / beta + T0) + Lat * f_l
    # print(h)
    # print(h_s)
    # print(h_l)

    for j in range(0, Ny):
        for i in range(0, Nx):
            if h[i, j] < h_s:
                f_l[i, j] = 0
            elif h[i, j] > h_l:
                f_l[i, j] = 1
            else:
                f_l[i, j] = (h[i, j] - h_s) / (h_l - h_s)
                # print(f_l[i, j])

                # x = 0
                # while True:
                #     x += 1
                #     h[i, j] = c_p * (T_dim[i, j] / beta + T0) + Lat * f_l[i, j]
                #     f_l_new = (h[i, j] - h_s) / (h_l - h_s)
                #     if x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                #         print(h[i, j])
                #         print(f_l_new)
                #
                #     if np.abs(f_l[i, j] - f_l_new) < 0.05:
                #         print('no')
                #         f_l[i, j] = f_l_new
                #         break
                #
                #     f_l[i, j] = f_l_new

    return f_l


# Buoyancy force
F_buoy = - T_dim[:, :, None] * g_sim

# Initialize equilibrium function
f_plus, f_minus, Si = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s, F_buoy)
f_i = f_plus + f_minus

start = time.time()

for t in range(Nt):
    # Buoyancy force
    F_buoy = - T_dim[:, :, None] * g_sim

    # Calculate macroscopic quantities
    rho_sim = np.sum(f_plus, axis=2)
    ux = f_l * (np.sum(f_minus[:, :] * c_i[:, 0], axis=2) / rho_sim + F_buoy[:, :, 0] / 2)
    uy = f_l * (np.sum(f_minus[:, :] * c_i[:, 1], axis=2) / rho_sim + F_buoy[:, :, 1] / 2)

    # Temperature
    T_dim = temperature(T_dim, alpha_sim, ux, uy, T_BC_C, T_BC_H)

    f_l = enthalpy(T_dim, h_s, h_l, f_l)

    # New equilibrium distribution
    f_eq_plus, f_eq_minus, Si = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s, F_buoy)

    # Collision step
    f_star = f_i - (f_plus - f_eq_plus) / tau_plus - (f_minus - f_eq_minus) / tau_minus + Si

    # Streaming step
    f_i = streaming(Nx, Ny, f_i, f_star)
    f_plus, f_minus = decompose_f_i(f_i)

    if t % 2500 == 0:
        # Heatmaps
        plt.figure(2)
        plt.clf()
        plt.imshow(f_l.T, cmap=cm.Blues, origin='lower')
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Liquid fraction, left wall at $T={T_C}K$, right wall at $T={T_H}K$')
        plt.colorbar()
        plt.savefig(f"Figures/hsource_sq_cav/heatmap_fl_time{Time}_{t%2500}_highres.png")

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

## Vector plot
# plt.figure(np.int(t/200)+1)
x = np.linspace(dx, L-dx, len(ux))
y = np.linspace(dx, H-dx, len(uy))
# fig = ff.create_streamline(x, y, ux.T, uy.T)
# fig.show()
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
# plt.legend('Velocity vector')
# plt.savefig("Figures/hsource_sq_cav/arrowplot_temp" + str(t-2) + ".png")

# # Vector plot
# plt.figure(np.int(t/200)+2, dpi=300)
# plt.quiver(Cu*ux.T, Cu*uy.T)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u$ in pipe with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# # plt.legend('Velocity vector')
# plt.savefig(f"Figures/hsource_sq_cav/arrowplot_time{Time}_test.png")
#
# Heatmaps
plt.figure(2)
plt.clf()
plt.imshow(f_l.T, cmap=cm.Blues, origin='lower')
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'Liquid fraction, left wall at $T={T_H}K$, right wall at $T={T_C}K$')
plt.colorbar()
plt.savefig(f"Figures/hsource_sq_cav/heatmap_fl_time{Time}_test_highres.png")
# # Heatmaps
# plt.figure(2)
# plt.clf()
# plt.imshow(Cu*ux.T, cmap=cm.Blues, origin='lower')
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u_x$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_sq_cav/heatmap_ux_time{Time}_test.png")
#
# plt.figure(3)
# plt.clf()
# plt.imshow(np.flip(Cu*uy, axis=1).T, cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$u_y$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_sq_cav/heatmap_uy_time{Time}_test.png")
#
# plt.figure(5)
# plt.clf()
# plt.imshow(np.flip(rho_sim, axis=1).T, cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'$\\rho$ in cavity with left wall at $T={T_H}K$ and right wall at $T={T_C}K$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_sq_cav/heatmap_rho_time{Time}_test.png")
#
# ## Temperature heatmap
# plt.figure(4)
# plt.clf()
# plt.imshow(np.flip(T.T, axis=0), cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('$T$ in cavity with left wall at $T=298K$ and right wall at $T=288K$. No flow.')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_sq_cav/heatmap_T_time{Time}_test.png")
#
# ## Temperature line plot
# plt.figure(0)
# plt.plot(x, T[:, 0])
# plt.xlabel('$x$ (m)')
# plt.ylabel('$T$ (K)')
# plt.title('$T$ in cavity with left wall at $T=298K$ and right wall at $T=288K$. 1D cross section.')
# plt.savefig(f"Figures/hsource_sq_cav/lineplot_T_adv_newBCs_time{Time}" + str() + ".png")
