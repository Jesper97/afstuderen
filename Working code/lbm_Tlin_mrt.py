import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streaming_funcs as str
import sys
import time
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
path_name = "/Users/Jesper/Documents/MEP/Code/Working code/FiguresLDC/"

# Physical parameters
Time = 1000         # (s)
L = 0.01             # Length of cavity (m)
H = 0.714*L      # Height of cavity (m)
g_phys = 9.81            # Gravitational acceleration (m/s^2)
rho_phys = 6.093e3      # Density (kg/m^3)
lbda_phys = 33           # Thermal conductivity (W/m K)
mu_phys = 1.81e-3        # Dynamic viscosity (Ns/m^2)
nu_phys = mu_phys / rho_phys      # Kinematic viscosity (m^2/s)
beta_phys = 1.2e-4       # Thermal expansion (1/K)
Lat_phys = 8.016e5       # Latent heat (J/kg)
cp_phys = 381           # Specific heat (J/(kgK))
alpha_phys = lbda_phys / (rho_phys * cp_phys)     # Thermal diffusivity (m^2/s)
Tm_phys = 302.8          # Melting point (K)
g_vec = g_phys * np.array([0, -1]) * 0

# Setup parameters
T0_phys = 302.67
TH_phys = 305
TC_phys = 301.3
epsilon = 0.01 * (TH_phys - Tm_phys)

umax = 0.1
bc_value = np.array([[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

# LBM parameters
w0 = 4/9
ws = 1/9
wd = 1/36
w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd], dtype=np.float32)
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
cs = 1 / np.sqrt(3)

# Dimensionless parameters
Pr = nu_phys / alpha_phys
print("Pr =", Pr)

# Simulation parameters
tau = 0.53
tau_inv = 1 / tau
Nx = 40
Ny = 3  #np.int(0.714*Nx)
rho = 1
nu = cs * (tau - 1/2)

# Calculate dependent parameters
alpha = nu / Pr

# Calculate conversion coefficients
dx = L / Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_phys
print("dx =", dx, "dt =", dt)
Cu = dx / dt
Cg = dx / dt**2
Crho = rho_phys / rho
Ch = dx**2 / dt**2
Ccp = Ch * beta_phys
Clbda = dx**4 / dt**3 * beta_phys * Crho

# Number of steps
Nt = np.int(Time/dt)

# Initial conditions
dim = (Nx, Ny)
rho = rho * np.ones(dim)  # Simulation density
T = np.zeros((Nx+2, Ny+2))     # Dimensionless simulation temperature
fL = np.zeros(dim)                # Liquid fraction
h = (cp_phys / Ccp) * np.zeros(dim)        # Enthalpy
c_app = cp_phys / Ccp * np.ones(dim)
Lat = Lat_phys / Ch

cs = cp_phys / Ccp
cl = cp_phys / Ccp
Ts = ((Tm_phys - epsilon) - T0_phys) * beta_phys
Tl = ((Tm_phys + epsilon) - T0_phys) * beta_phys
lbda = lbda_phys / Clbda

hs = cs * Ts
hl = hs + Lat + (cs + cl) / 2 * (Tl - Ts)

# Temperature BCS
TH = beta_phys * (TH_phys - T0_phys)
TC = beta_phys * (TC_phys - T0_phys)

# 1D Stefan problem
xi = 0.02286148

# Collision operators
M_rho = np.ones(9)
M_e = np.array([-4, -1, -1, -1, -1, 2, 2, 2, 2])
M_eps = np.array([4, -2, -2, -2, -2, 1, 1, 1, 1])
M_jx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
M_qx = np.array([0, -2, 0, 2, 0, 1, -1, -1, 1])
M_jy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
M_qy = np.array([0, 0, -2, 0, 2, 1, 1, -1, -1])
M_pxx = np.array([0, 1, -1, 1, -1, 0, 0, 0, 0])
M_pyy = np.array([0, 0, 0, 0, 0, 1, -1, 1, -1])
M = np.array([M_rho, M_e, M_eps, M_jx, M_qx, M_jy, M_qy, M_pxx, M_pyy])
M_inv = np.dot(M.T, np.linalg.inv(np.dot(M, M.T)))

s0 = 0
s1 = 1.4
s2 = 1.4
s3 = 0
s7 = 1 / tau
s4 = 1.2 #8 * ((2 - s7) / (8 - s7))
s5 = 0
s6 = s4
s8 = s7
S = np.diag(np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8]))

MSM = np.dot(M_inv, np.dot(S, M))

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


@njit
def f_eq(rho, vel):
    feq = np.zeros((Nx, Ny, 9))
    uv = vel[:, :, 0]*vel[:, :, 0] + vel[:, :, 1]*vel[:, :, 1]
    for k in range(9):
        uc = e[k, 0] * vel[:, :, 0] + e[k, 1] * vel[:, :, 1]
        feq[:, :, k] = w[k] * rho * (1.0 + 3.0 * uc + 4.5 * uc*uc - 1.5 * uv)

    return feq


def initialize(g):
    rho = np.ones((Nx, Ny))
    vel = np.zeros((Nx, Ny, 2))

    f_new = f_eq(rho, vel)
    f_old = f_new.copy()

    Si = np.zeros((Nx, Ny, 9))
    F = rho[:, :, None] * g     # Need shape and values of rho, not physical

    T = np.zeros((Nx+2, Ny+2))

    return vel, rho, f_new, f_old, Si, F, T


@njit
def streaming(rho, f_new, f_old):
    f = str.fluid(Nx, Ny, e, f_new, f_old)
    f = str.left_right_wall(Nx, Ny, f, f_old)
    f = str.top_bottom_wall(Nx, Ny, f, f_old, w, rho, bc_value)
    f = str.bottom_corners(Nx, Ny, f, f_old)
    f = str.top_corners(Nx, Ny, f, f_old, w, rho, bc_value)

    return f


def collision(r, u, f_old, Si):
    feq = f_eq(r, u)                                                    # Equilibrium
    f_new = f_old - np.einsum('ij,klj->kli', MSM, f_old - feq) + Si     # Collision

    return f_new


@njit
def forcing(vel, g, Si, F, T):
    F[:, :, 0] = - T[1:-1, 1:-1] * g[0]
    F[:, :, 1] = - T[1:-1, 1:-1] * g[1]
    uF = vel[:, :, 0] * F[:, :, 0] + vel[:, :, 1] * F[:, :, 1]                             # Inner product of u with F
    for i in range(9):
        Si[:, :, i] = (1 - tau_inv/2) * w[i] * (3 * (F[:, :, 0]*e[i, 0]+F[:, :, 1]*e[i, 1]) + 9 * (vel[:, :, 0]*e[i, 0]+vel[:, :, 1]*e[i, 1]) * (F[:, :, 0]*e[i, 0]+F[:, :, 1]*e[i, 1]) - 3 * uF)

    return Si, F


@njit
def temperature(T_iter, h_old, c_app_iter, f_l_old, ux, uy, rho, T_dim_C, T_dim_H, t):
    T_new = np.zeros((Nx+2, Ny+2))
    l_relax = 0.1

    def energy_eq(i, j, T, ux, uy, rho, c_app, h, h_old):
        im = i - 1
        jm = j - 1
        ip = i + 1
        jp = j + 1
        a_app = lbda / (c_app[im, jm] * rho[im, jm])
        T_new = T[i, j] * (1 - 6 * a_app) + T[i+1, j] * (-ux[im, jm] + 2 * a_app) + T[im, j] * (ux[im, jm] + 2 * a_app) + \
            T[i, jp] * (-uy[im, jm] + 2 * a_app) + T[i, jp] * (uy[im, jm] + 2 * a_app) + \
            T[ip, jp] * (ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) + \
            T[im, jp] * (-ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) + \
            T[ip, jm] * (ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + \
            T[im, jm] * (-ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) - (h[im, jm] - h_old[im, jm]) / c_app[im, jm]

        return T_new

    f_l_iter = -np.ones(f_l_old.shape)  # Initialize

    h_iter = h_old.copy()
    f_l = f_l_old.copy()

    dT = Tl - Ts
    dTdh = dT / (hl - hs)
    cs_LatdT = cs + (Lat / dT)
    dcdT = (cl - cs) / dT

    n_iter = 1
    while True:
        for j in range(1, Ny+1):
            for i in range(1, Nx+1):
                T_new[i, j] = energy_eq(i, j, T_iter, ux, uy, rho, c_app_iter, h_iter, h_old)

        h_new = h_iter + l_relax * c_app_iter * (T_new[1:-1, 1:-1] - T_iter[1:-1, 1:-1])

        for j in range(1, Ny+1):
            for i in range(1, Nx+1):
                im = i - 1
                jm = j - 1
                if h_new[im, jm] < hs:
                    T_new[i, j] = h_new[im, jm] / cs
                elif h_new[im, jm] > hl:
                    T_new[i, j] = Tl + (h_new[im, jm] - hl) / cl
                else:
                    T_new[i, j] = Ts + (h_new[im, jm] - hs) * dTdh

                if T_new[i, j] < Ts:
                    c_app[im, jm] = cs
                    f_l[im, jm] = 0
                elif T_new[i, j] > Tl:
                    c_app[im, jm] = cl
                    f_l[im, jm] = 1
                else:
                    c_app[im, jm] = cs_LatdT + (T_new[i, j] - Ts) * dcdT
                    f_l[im, jm] = (T_new[i, j] - Ts) / dT

        # # Ghost nodes
        T_new[1:-1, 0] = 21/23 * T_new[1:-1, 1] + 3/23 * T_new[1:-1, 2] - 1/23 * T_new[1:-1, 3]         # Neumann extrapolation on lower boundary
        T_new[1:-1, -1] = 21/23 * T_new[1:-1, -2] + 3/23 * T_new[1:-1, -3] - 1/23 * T_new[1:-1, -4]     # Neumann extrapolation on upper boundary
        T_new[-1, :] = 21/23 * T_new[-2, :] + 3/23 * T_new[-3, :] - 1/23 * T_new[-4, :]               # Neumann extrapolation on right boundary
        T_new[0, :] = 16/5 * T_dim_H - 3 * T_new[1, :] + T_new[2, :] - 1/5 * T_new[3, :]               # Dirichlet extrapolation on left boundary
        # T_new[-1, :] = 16/5 * T_C - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]           # Dirichlet extrapolation on right boundary

        if np.any(np.abs(f_l - f_l_iter)) < 1e-5:
            break
        else:
            T_iter = T_new.copy()
            h_iter = h_new.copy()
            c_app_iter = c_app.copy()
            f_l_iter = f_l.copy()

        n_iter += 1

    if t % 1000 == 0:
        print(t)

    return T_new, h_new, c_app, f_l


@njit
def moment_update(rho, vel, f_old, f_new, F):
    rho[:, :] = np.sum(f_new, axis=2)
    vel[:, :, 0] = (f_new[:, :, 1] + f_new[:, :, 5] + f_new[:, :, 8] - (f_new[:, :, 3] + f_new[:, :, 6] + f_new[:, :, 7])) / rho + F[:, :, 0] / 2
    vel[:, :, 1] = (f_new[:, :, 2] + f_new[:, :, 5] + f_new[:, :, 6] - (f_new[:, :, 4] + f_new[:, :, 7] + f_new[:, :, 8])) / rho + F[:, :, 1] / 2

    f_old = f_new

    return rho, vel, f_old


def solve(h, c_app, fL):
    vel, rho, f_str, f_old, Si, F, T = initialize(g_vec)

    for t in range(Nt):
        # Si, F = forcing(vel, g_vec, Si, F, T)
        # rho, vel, f_old = moment_update(rho, vel, f_old, f_str, F)
        T, h, c_app, fL = temperature(T, h, c_app, fL, vel[:, :, 0], vel[:, :, 1], rho, TC, TH, t)
        # f_col = collision(rho, vel, f_old, Si)
        # f_str = streaming(rho, f_old, f_col)

        # if i % 10000 == 0:
        #     print(t)

    return vel


start = time.time()

u = solve(h, c_app, fL)

stop = time.time()
print(stop-start)

# Compare results with literature
y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 3))
x_ref, v_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 9))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(u[Nx // 2, :, 0] / umax, np.linspace(0, 1.0, Nx), 'b-', label='LBM')
axes.plot(u_ref, y_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig(path_name + "ux_Re1000_test_mrt.png")

plt.clf()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(np.linspace(0, 1.0, Nx), u[:, Nx // 2, 1] / umax, 'b-', label='LBM')
axes.plot(x_ref, v_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig(path_name + "uy_Re1000_test_mrt.png")
