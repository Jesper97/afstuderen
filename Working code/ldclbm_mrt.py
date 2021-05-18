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

steps = 50000
umax = 0.1
Nx = 128  # by convention, dx = dy = dt = 1.0 (lattice units)
Ny = Nx
nu = 0.0128
tau = 3.0 * nu + 0.5
tau_inv = 1.0 / tau
bc_value = np.array([[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

w0 = 4/9
ws = 1/9
wd = 1/36

w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd], dtype=np.float32)
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)

# Dimensionless parameters
Pr = 0.021

# Forces
g = 9.81 * np.array([0, -1])

# Material
cp = 381
Lat = 8.016e5
beta = 1.2e-4
lbda = 33
rho_ph = 6.093e3
mu_ph = 1.81e-3
nu_ph = mu_ph / rho_ph
alpha_ph = lbda / (rho_ph * cp)

# Temperature
Tm = 302.8          # Melting point (K)
T0 = 302.67          # Starting temperature (K)
TH = 305           # Hot wall temperature (K)
TC = 301.3         # Cold wall temperature (K)
epsilon = 0.05 * (TH - Tm) # 0.05 * (T_H - T_C)  # Width mushy zone (K)

##
alpha = Pr / nu


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
s1 = 1.64
s2 = 1.2
s3 = 0
s7 = 1 / tau
s4 = 8 * ((2 - s7) / (8 - s7))
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


# @njit
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
def temperature(T_old):
    T_new = np.zeros((Nx+2, Ny+2))
    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            while True:
                n_iter = 1
                while True:
                    T_new[i, j] = T_old[i, j] * (1 - 6 * alpha) - 



@njit
def moment_update(rho, vel, f_old, f_new, F):
    rho[:, :] = np.sum(f_new, axis=2)
    vel[:, :, 0] = (f_new[:, :, 1] + f_new[:, :, 5] + f_new[:, :, 8] - (f_new[:, :, 3] + f_new[:, :, 6] + f_new[:, :, 7])) / rho + F[:, :, 0] / 2
    vel[:, :, 1] = (f_new[:, :, 2] + f_new[:, :, 5] + f_new[:, :, 6] - (f_new[:, :, 4] + f_new[:, :, 7] + f_new[:, :, 8])) / rho + F[:, :, 1] / 2

    f_old = f_new

    return rho, vel, f_old


def solve():
    vel, rho, f, f_old, Si, F, T = initialize(g)

    for i in range(steps):
        f_col = collision(rho, vel, f_old, Si)
        f_str = streaming(rho, f_old, f_col)
        rho, vel, f_old = moment_update(rho, vel, f_old, f_str, F)
        Si, F = forcing(vel, g, Si, F, T)

        if i % 10000 == 0:
            print(i)

    return vel


start = time.time()

u = solve()

stop = time.time()
print(stop-start)

# Compare results with literature
y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 2))
x_ref, v_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 8))

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
