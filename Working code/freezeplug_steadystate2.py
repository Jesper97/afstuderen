import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streaming_funcs_plug as str
import sys
import time
from matplotlib import cm
from numba import njit, prange
from numpy import zeros, empty, ones, einsum, any
from numpy import abs as npabs
from numpy import sum as npsum

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


# Domain parameters
Time = 6000
W_wall = 0.02
L_cooled = 0.2
W = 0.2 + 2 * W_wall
L = 0.3
g_phys = 9.81
g_vec_phys = np.array([0, -g_phys])

# Rotation of the domain
phi = 90        # Degrees
phi_ccw = 2 * np.pi * (1 - phi / 360)
rotation_mat = np.array([[np.cos(phi_ccw), -np.sin(phi_ccw)], [np.sin(phi_ccw), np.cos(phi_ccw)]])
g_vec_p_rot = np.dot(rotation_mat, g_vec_phys)

# Material parameters salt
rho_salt_p = 4480
cp_salt_sol_p = 815
cp_salt_liq_p = 1000
lbda_salt_p = 1.5
mu_salt_p = 9.1e-3
nu_salt_p = 2.03e-6
alpha_salt_liq_p = 4.09e-7
alpha_salt_sol_p = 3.42e-7
beta_salt_p = 2.79e-4
Lat_salt_p = 1.59e5
Tm_salt_p = 841

# Material parameters Hastelloy-N
# rho_HN_p = 8860
cp_HN_p = 578
lbda_HN_p = 23.6
# mu_HN_p = 9.1e-3
# nu_HN_p = 2.03-6
alpha_HN_p = 4.61e-6
beta_HN_p = 0
Lat_HN_p = 1e10
Tm_HN_p = 1372

# LBM parameters
q = 9
w0 = 4/9
ws = 1/9
wd = 1/36
w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd])
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
cs = 1/np.sqrt(3)

# Temperatures
T0_p = 910
Tsub_p = 15
TH_p = 923
TC_p = Tm_salt_p - Tsub_p
epsilon = 0.01 * (TH_p - TC_p)

# Dimensionless numbers
Pr = nu_salt_p / alpha_salt_liq_p
Ra = g_phys * beta_salt_p * (TH_p - TC_p) * (W-2*W_wall)**3 / (nu_salt_p * alpha_salt_liq_p)
Ste = cp_salt_liq_p * (TH_p - TC_p) / Lat_salt_p
# FoSte_t = Ste * alpha_phys / L**2

# Simulation parameters
l_relax = 1
tau = 0.52
tau_inv = 1/tau
Nx = 240
Ny = np.int(W / L * Nx)
rho0 = 1
nu = cs**2 * (tau - 1/2)
alpha_salt = nu / Pr

# Parameter arrays
dim = (Nx, Ny)
idx_boundary = np.int(W_wall / W * Ny)
idx_cooled = np.int(L_cooled / L * Nx)

cp_p = cp_salt_sol_p * np.ones(dim)
cp_p[:, 0:idx_boundary] = cp_HN_p
cp_p[:, -idx_boundary:] = cp_HN_p

alpha_p = alpha_salt_sol_p * np.ones(dim)
alpha_p[:, 0:idx_boundary] = alpha_HN_p
alpha_p[:, -idx_boundary:] = alpha_HN_p

# Calculate dependent parameters
dx = L / Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_salt_p
Cu = dx / dt
Cg = dx / dt**2
Crho = rho_salt_p / rho0
Ch = dx**2 / dt**2
Ccp = Ch * beta_salt_p
Clbda = Crho * dx**4 / dt**3 * beta_salt_p
Calpha = alpha_salt_liq_p / alpha_salt

Nt = np.int(Time/dt)
Nresponse = np.int(Nt/40 - 5)

# Initial conditions
cp_sol = cp_salt_sol_p / Ccp
cp_liq = cp_salt_liq_p / Ccp
cp = cp_p / Ccp
cp[idx_cooled:, idx_boundary:Ny-idx_boundary] = cp_liq

alpha_HN = alpha_HN_p / Calpha
alpha_sol = alpha_salt_sol_p / Calpha
alpha_liq = alpha_salt_liq_p / Calpha
alpha = alpha_p / Calpha
alpha[idx_cooled:, idx_boundary:Ny-idx_boundary] = alpha_liq

B = np.ones(dim)
h = np.zeros(dim)
Lat = Lat_salt_p / Ch

g_vec = g_vec_p_rot / Cg

# Temperature BCS
TH = beta_salt_p * (TH_p - T0_p)
TC = beta_salt_p * (TC_p - T0_p)

s0 = 0
s1 = 1.64
s2 = 1.2
s3 = 0
s4 = 8 * ((2 - tau_inv) / (8 - tau_inv))
s5 = 0
s6 = s4
s7 = tau_inv
s8 = s7
S = np.diag(np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8]))

M0 = np.ones(q)
M1 = np.array([-4, -1, -1, -1, -1, 2, 2, 2, 2])
M2 = np.array([4, -2, -2, -2, -2, 1, 1, 1, 1])
M3 = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
M4 = np.array([0, -2, 0, 2, 0, 1, -1, -1, 1])
M5 = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
M6 = np.array([0, 0, -2, 0, 2, 1, 1, -1, -1])
M7 = np.array([0, 1, -1, 1, -1, 0, 0, 0, 0])
M8 = np.array([0, 0, 0, 0, 0, 1, -1, 1, -1])

M = np.array([M0, M1, M2, M3, M4, M5, M6, M7, M8])
M_inv = np.linalg.inv(M)

MSM = np.dot(M_inv, np.dot(S, M))

material = "LiF-ThF4"
print("Simulating", material)
print("Pr =", Pr)
print("Ra =", Ra)
print("Ste =", Ste)
print("dx =", dx, "dt =", dt)
print("Nodes:", Nx, "x", Ny)
print(f"{Nt} steps")
if alpha_salt > 1/6:
    print(f"Warning alpha = {np.round(alpha_salt, 2)}. Can cause stability or convergence issues.")
if alpha_HN > 1/6:
    print(f"Warning alpha = {np.round(alpha_HN, 2)}. Can cause stability or convergence issues.")

# CSV filenames
path_name = f"/Users/Jesper/Documents/MEP/Code/Working code/Figures/Freeze Plug/tau=0.51/"
suffix = f"freeze_plug_W=0.2_tau={tau}_N={Nx}x{Ny}.png"
csv_path = f"/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.51/"
csv_file = f"freeze_plug_W=0.2_tau={tau}_N={Nx}x{Ny}"
print(suffix)


def initialize(g):
    ##### From start
    # rho = rho0 * ones((Nx, Ny))
    # vel = zeros((Nx, Ny, 2))
    #
    # f_new = f_eq(rho, vel)
    #
    # T = zeros((Nx+2, Ny+2))
    #
    # Si = zeros((Nx, Ny, q))
    # F = - T[1:-1, 1:-1, None] * rho[:, :, None] * g
    #
    # T[1:idx_cooled+1, :] = (Tm_salt_p - Tsub_p - T0_p) * beta_salt_p
    #
    # fL = np.zeros(dim)
    # fL[idx_cooled:, idx_boundary:Ny-idx_boundary] = 1

    #### From csv
    rho = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.502/rho_freeze_plug_W=0.2_constantT_tau=0.502_N=240x192_t=2000.0.csv', delimiter=',')

    vel = zeros((Nx, Ny, 2))
    ux = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.502/ux_freeze_plug_W=0.2_constantT_tau=0.502_N=240x192_t=2000.0.csv', delimiter=',')
    uy = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.502/uy_freeze_plug_W=0.2_constantT_tau=0.502_N=240x192_t=2000.0.csv', delimiter=',')
    vel[:, :, 0] = ux.T / Cu
    vel[:, :, 1] = np.rot90(np.rot90(np.rot90(uy))) / Cu

    fL = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.502/fL_freeze_plug_W=0.2_constantT_tau=0.502_N=240x192_t=2000.0.csv', delimiter=',')
    fL = fL.T

    T = zeros((Nx+2, Ny+2))
    F = - T[1:-1, 1:-1, None] * rho[:, :, None] * g
    T_p = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/tau=0.502/T_freeze_plug_W=0.2_constantT_tau=0.502_N=240x192_t=2000.0.csv', delimiter=',')
    T[1:-1, 1:-1] = beta_salt_p * (T_p.T - T0_p)

    f_new = f_eq(rho, vel)
    Si = zeros((Nx, Ny, q))

    # easy_view("ux", vel[:, :, 0])
    # easy_view("uy", vel[:, :, 1])
    # easy_view("fL", fL)
    # easy_view("T", T)

    #####
    # t = 0
    # T_phys = T / beta_salt_p + T0_p
    # TH_phys = TH / beta_salt_p + T0_p
    # ux_phys = vel[:, :, 0] * Cu     # * L / alpha_phys
    # uy_phys = vel[:, :, 1] * Cu     # * L / alpha_phys
    #
    # # Liquid fraction
    # plt.figure()
    # plt.imshow(fL.T, cmap=cm.RdBu, origin='lower', aspect=1.0)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $f_L$, $t={np.int(t/Nt*Time)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_fl_t={np.int(t/Nt*Time)}" + suffix)
    #
    # # Streamlines velocity
    # uy_plot = np.rot90(uy_phys)
    # ux_plot = ux_phys.T
    #
    # # Velocities
    # plt.figure()
    # plt.clf()
    # plt.imshow(uy_plot, cmap=cm.Blues)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes) ')
    # plt.title(f'Octadecane \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}" + suffix)
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(ux_plot, cmap=cm.Blues, origin='lower')
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}" + suffix)
    #
    # # Temperature heatmap
    # cmap = cm.get_cmap('PiYG', 11)
    # plt.figure()
    # plt.clf()
    # plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cmap)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_T_t={np.round(t/Nt*Time, decimals=2)}" + suffix)
    #
    # rho[:, 0:idx_boundary] = 1
    # rho[:, -idx_boundary:] = 1
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $\\rho$, left wall at $T={TH_phys}K$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    return vel, rho, f_new, Si, F, T, fL


@njit
def streaming(f_old):
    f_new = empty((Nx, Ny, 9))
    f = str.fluid(idx_boundary, Nx, Ny, e, f_new, f_old)
    f = str.left_right_wall(idx_boundary, Nx, Ny, f, f_old)
    f = str.top_bottom_wall(idx_boundary, Nx, Ny, f, f_old)
    f = str.bottom_corners(idx_boundary, Nx, Ny, f, f_old)
    f = str.top_corners(idx_boundary, Nx, Ny, f, f_old)

    return f


@njit(parallel=True)
def f_eq(rho, vel):
    feq = zeros((Nx, Ny, q))
    for j in prange(Ny):
        for i in prange(Nx):
            for k in prange(q):
                uc = e[k, 0] * vel[i, j, 0] + e[k, 1] * vel[i, j, 1]
                feq[i, j, k] = w[k] * rho[i, j] * (1.0 + 3.0 * uc + 4.5 * uc*uc -
                                                   1.5 * (vel[i, j, 0]*vel[i, j, 0] + vel[i, j, 1]*vel[i, j, 1]))

    return feq


@njit
def collision_speedup(Omegaf, fL, f, Si):
    Bi = zeros((Nx, Ny, 9))
    B = (1 - fL) * (tau - 1/2) / (fL + tau - 1/2)
    for k in range(9):
        Bi[:, :, k] = B

    return B, f - (1 - Bi) * Omegaf + Bi * (f[:, :, opp] - f) + Si * (1 - Bi)


def collision(r, u, f_old, Si, fL):
    Omegaf = einsum('ij,klj->kli', MSM, f_old - f_eq(r, u) - Si/2)
    return collision_speedup(Omegaf, fL, f_old, Si)


@njit(parallel=True)
def forcing(vel, g, T):
    Si = zeros((Nx, Ny, q))
    F = zeros((Nx, Ny, q))
    for j in prange(idx_boundary, Ny - idx_boundary):
        for i in prange(Nx):
            ip = i + 1
            jp = j + 1
            F[i, j, 0] = - T[ip, jp] * g[0] * rho0
            F[i, j, 1] = - T[ip, jp] * g[1] * rho0
            for k in prange(q):
                eF = F[i, j, 0]*e[k, 0]+F[i, j, 1]*e[k, 1]
                Si[i, j, k] = w[k] * (3 * eF + 9 * (vel[i, j, 0]*e[k, 0]+vel[i, j, 1]*e[k, 1]) * eF -
                                      3 * (vel[i, j, 0]*F[i, j, 0]+vel[i, j, 1]*F[i, j, 1]))

    return Si, F


@njit(parallel=True)
def temperature(T_old, f_l_old, cp, alpha, ux, uy, t, TC, TH):
    T_new = np.zeros((Nx+2, Ny+2))
    f_l_new = np.zeros((Nx, Ny))
    l_relax = 1

    Ts = ((Tm_salt_p - epsilon) - T0_p) * beta_salt_p
    Tl = ((Tm_salt_p + epsilon) - T0_p) * beta_salt_p
    h_s = cp_salt_sol_p * Ts

    h_l = h_s + Lat + (cp_salt_sol_p + cp_salt_liq_p) / 2 * (Tl - Ts)

    f_l_iter = f_l_old.copy()

    for j in prange(idx_boundary+1, Ny-idx_boundary+1):
        for i in prange(1, Nx+1):
            im, jm, ip, jp = i-1, j-1, i+1, j+1
            constant_terms = T_old[i, j] - ux[i-1, j-1] * (T_old[i+1, j] - T_old[i-1, j] - 1/4 * (T_old[i+1, j+1] - T_old[i-1, j+1] + T_old[i+1, j-1] - T_old[i-1, j-1]))\
                             - uy[i-1, j-1] * (T_old[i, j+1] - T_old[i, j-1] - 1/4 * (T_old[i+1, j+1] - T_old[i+1, j-1] + T_old[i-1, j+1] - T_old[i-1, j-1]))\
                             + alpha[i-1, j-1] * (2 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1]) - 1/2 * (T_old[i+1, j+1] + T_old[i-1, j+1] + T_old[i-1, j-1] + T_old[i+1, j-1]) - 6 * T_old[i, j])

            while True:
                n_iter = 1
                while True:
                    T_new[i, j] = constant_terms - Lat / cp[i-1, j-1] * (f_l_iter[i-1, j-1] - f_l_old[i-1, j-1])

                    h = cp[i-1, j-1] * T_new[i, j] + f_l_iter[i-1, j-1] * Lat

                    if h < h_s:
                        f_l_new[i-1, j-1] = 0
                    elif h > h_l:
                        f_l_new[i-1, j-1] = 1
                    else:
                        f_l_new[i-1, j-1] = (h - h_s) / (h_l - h_s)

                    f_l_new[i-1, j-1] = min(max(f_l_new[i-1, j-1], 0), 1)

                    if np.abs(f_l_new[i-1, j-1] - f_l_iter[i-1, j-1]) < 1e-6 and (n_iter >= 3):
                        break
                    elif (n_iter > 1000) and (l_relax == 1):
                        print('yes', t)
                        l_relax = 0.1
                        break
                    else:
                        f_l_iter[i-1, j-1] = f_l_new[i-1, j-1]

                    n_iter += 1

                if np.abs(f_l_new[i-1, j-1] - f_l_iter[i-1, j-1]) < 1e-5:
                    break
                else:
                    continue

            if f_l_new[im, jm] == 1:
                cp[im, jm] = cp_liq
                alpha[im, jm] = alpha_liq
            elif f_l_new[im, jm] == 0:
                cp[im, jm] = cp_sol
                alpha[im, jm] = alpha_sol
            else:
                cp[im, jm] = f_l_new[im, jm] * cp_liq + (1 - f_l_new[im, jm]) * cp_sol
                alpha[im, jm] = f_l_new[im, jm] * alpha_liq + (1 - f_l_new[im, jm]) * alpha_sol

    for j in prange(1, idx_boundary+1):
        for i in prange(1, Nx+1):
            T_new[i, j] = T_old[i, j] - ux[i-1, j-1] * (T_old[i+1, j] - T_old[i-1, j] - 1/4 * (T_old[i+1, j+1] - T_old[i-1, j+1] + T_old[i+1, j-1] - T_old[i-1, j-1]))\
                          - uy[i-1, j-1] * (T_old[i, j+1] - T_old[i, j-1] - 1/4 * (T_old[i+1, j+1] - T_old[i+1, j-1] + T_old[i-1, j+1] - T_old[i-1, j-1]))\
                          + alpha[i-1, j-1] * (2 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1]) - 1/2 * (T_old[i+1, j+1] + T_old[i-1, j+1] + T_old[i-1, j-1] + T_old[i+1, j-1]) - 6 * T_old[i, j])

    for j in prange(Ny-idx_boundary+1, Ny+1):
        for i in prange(1, Nx+1):
            T_new[i, j] = T_old[i, j] - ux[i-1, j-1] * (T_old[i+1, j] - T_old[i-1, j] - 1/4 * (T_old[i+1, j+1] - T_old[i-1, j+1] + T_old[i+1, j-1] - T_old[i-1, j-1]))\
                          - uy[i-1, j-1] * (T_old[i, j+1] - T_old[i, j-1] - 1/4 * (T_old[i+1, j+1] - T_old[i+1, j-1] + T_old[i-1, j+1] - T_old[i-1, j-1]))\
                          + alpha[i-1, j-1] * (2 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1]) - 1/2 * (T_old[i+1, j+1] + T_old[i-1, j+1] + T_old[i-1, j-1] + T_old[i+1, j-1]) - 6 * T_old[i, j])

    # Ghost nodes
    T_new[0, :] = 21/23 * T_new[1, :] + 3/23 * T_new[2, :] - 1/23 * T_new[3, :]                                                         # Neumann extrapolation on left boundary
    T_new[-1, :] = 16/5 * TH - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]                                                     # Dirichlet extrapolation on right boundary
    T_new[1:idx_cooled, -1] = 16/5 * TC - 3 * T_new[1:idx_cooled, -2] + T_new[1:idx_cooled, -3] - 1/5 * T_new[1:idx_cooled, -4]         # Dirichlet extrapolation on upper cold boundary
    T_new[1:idx_cooled, 0] = 16/5 * TC - 3 * T_new[1:idx_cooled, 1] + T_new[1:idx_cooled, 1] - 1/5 * T_new[1:idx_cooled, 1]             # Dirichlet extrapolation on lower cold boundary
    T_new[idx_cooled:-1, -1] = 21/23 * T_new[idx_cooled:-1, -2] + 3/23 * T_new[idx_cooled:-1, -3] - 1/23 * T_new[idx_cooled:-1, -4]     # Neumann extrapolation on upper adiab boundary
    T_new[idx_cooled:-1, 0] = 21/23 * T_new[idx_cooled:-1, 1] + 3/23 * T_new[idx_cooled:-1, 2] - 1/23 * T_new[idx_cooled:-1, 3]         # Neumann extrapolation on lower adiab boundary

    return T_new, f_l_new, cp, alpha


@njit(parallel=True)
def moment_update(f_new, F, B):
    rho = zeros((Nx, Ny))
    vel = zeros((Nx, Ny, 2))
    for j in prange(idx_boundary, Ny-idx_boundary):
        for i in prange(Nx):
            rho[i, j] = npsum(f_new[i, j, :])
            if B[i, j] != 0:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j] + F[i, j, 0] / (2 * rho0)
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j] + F[i, j, 1] / (2 * rho0)

    return rho, vel


@njit
def moment_plots(f_new, B):
    rho = zeros((Nx, Ny))
    vel = zeros((Nx, Ny, 2))
    for j in prange(idx_boundary, Ny-idx_boundary):
        for i in prange(Nx):
            rho[i, j] = npsum(f_new[i, j, :])
            if B[i, j] != 0:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j]
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j]

    return rho, vel


def outputs(f_str, T, fL, B, t):
    rho, vel = moment_plots(f_str, B)
    T_phys = T / beta_salt_p + T0_p
    TH_phys = TH / beta_salt_p + T0_p
    ux_phys = vel[:, :, 0] * Cu     # * L / alpha_phys
    uy_phys = vel[:, :, 1] * Cu     # * L / alpha_phys

    # Liquid fraction
    plt.figure()
    plt.imshow(fL.T, cmap=cm.RdBu, origin='lower', aspect=1.0)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Octadecane \n $f_L$, $t={np.int(t/Nt*Time)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_fl_t={np.int(t/Nt*Time)}" + suffix)

    # Streamlines velocity
    uy_plot = np.rot90(uy_phys)
    ux_plot = ux_phys.T

    # plt.clf()
    # plt.figure()
    # x = np.linspace(0, 1, Nx)
    # y = np.linspace(0, 1, Ny)
    # u = np.linspace(0, 1, 100)
    # g = np.meshgrid(u, u)
    # str_pts = list(zip(*(x.flat for x in g)))
    # plt.streamplot(x, y, ux_plot, np.flip(uy_plot, axis=0),
    #                linewidth    = 1.5,
    #                cmap         = 'RdBu_r',
    #                arrowstyle   = '-',
    #                start_points = str_pts,
    #                density      = 1)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.savefig(path_name + f"streamlines_u_t={np.round(t/Nt*Time, decimals=2)}" + suffix)
    # plt.close()

    # # Contour plots
    # X, Y = np.meshgrid(x, y)
    # plt.figure()
    # CS = plt.contour(X, Y, np.flip(uy_plot, axis=1))
    # plt.clabel(CS, inline=True)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.savefig(path_name + f"contour_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    # plt.figure()
    # plt.clf()
    # CS = plt.contour(X, Y, ux_plot)
    # plt.clabel(CS, inline=True)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.savefig(path_name + f"contour_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    # Velocities
    plt.figure()
    plt.clf()
    plt.imshow(uy_plot, cmap=cm.Blues)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes) ')
    plt.title(f'Octadecane \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}" + suffix)

    plt.figure()
    plt.clf()
    plt.imshow(ux_plot, cmap=cm.Blues, origin='lower')
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Octadecane \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}" + suffix)

    # Temperature heatmap
    cmap = cm.get_cmap('PiYG', 11)
    plt.figure()
    plt.clf()
    plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cmap)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Octadecane \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_T_t={np.round(t/Nt*Time, decimals=2)}" + suffix)

    # print("T_max", np.max(T_phys[1:-1, 1:-1]))
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $\\rho$, left wall at $T={TH_phys}K$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    # # Vector plot
    # plt.figure()
    # plt.quiver(ux_plot, np.flip(uy_plot, axis=1))
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Octadecane \n $u$ in pipe with left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # # plt.legend('Velocity vector')
    # plt.savefig(path_name + f"arrowplot_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    plt.close('all')

    # Save arrays to CSV-files
    np.savetxt(csv_path+"rho_"+csv_file+f"_t={np.round(t/Nt*Time)}.csv",    rho,       delimiter=",")
    np.savetxt(csv_path+"fL_"+csv_file+f"_t={np.round(t/Nt*Time)}.csv",     fL.T,      delimiter=",")
    np.savetxt(csv_path+"ux_"+csv_file+f"_t={np.round(t/Nt*Time)}.csv",     ux_plot,   delimiter=",")
    np.savetxt(csv_path+"uy_"+csv_file+f"_t={np.round(t/Nt*Time)}.csv",     uy_plot,   delimiter=",")
    np.savetxt(csv_path+"T_"+csv_file+f"_t={np.round(t/Nt*Time)}.csv",      T_phys.T,  delimiter=",")


def solve(B, cp, alpha):
    vel, rho, f_str, Si, F, T, fL = initialize(g_vec)
    T_old = T.copy()
    fL_old = fL.copy()

    for t in range(Nt):
        # t_p = t / Nt * Time
        # TH = (-0.0001 * t_p**2 + 0.5244 * t_p + 923 - T0_p) * beta_salt_p
        T, fL, cp, alpha = temperature(T, fL, cp, alpha, vel[:, :, 0], vel[:, :, 1], t, TC, TH)
        Si, F = forcing(vel, g_vec, T)
        rho, vel = moment_update(f_str, F, B)
        B, f_col = collision(rho, vel, f_str, Si, fL)
        f_str = streaming(f_col)
        # easy_view("cp", cp)

        if t % 2500 == 0:
            print(t)

            if t == 0:
                begin = time.time()
            if t == 10000:
                end = time.time()
                runtime = (end - begin) * Nt / 10000
                mins = np.round(runtime/60, 1)
                print("Estimated runtime:", mins, "minutes.")

        if (t % Nresponse == 0) and (t != 0):
            # easy_view("alpha", alpha)
            # easy_view('cp', cp)
            # easy_view('fL', fL )
            outputs(f_str, T, fL, B, t)

        if (t % Nt/2000 == 0) and (t > 10000):
            print("T diff", np.max(npabs(T - T_old)))
            print("fL diff", np.max(npabs(fL - fL_old)))

            if (npabs(T - T_old) < 1e-8).all():
                if (npabs(fL - fL_old) < 1e-8).all():
                    outputs(f_str, T, fL, B, t)
                    sys.exit("Convergence reached")
            if t % 50000 == 0:
                print("T convergence", np.max(npabs(T - T_old)))
                print("fL convergence", np.max(npabs(fL - fL_old)))

            T_old = T.copy()
            fL_old = fL.copy()


start = time.time()

solve(B, cp, alpha)

stop = time.time()
run_time = np.array([stop - start])
print(run_time[0])

np.savetxt(csv_path+"run_time"+csv_file+".csv", run_time, delimiter=",")
