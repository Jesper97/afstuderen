import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streaming_funcs as str
import sys
import time
from matplotlib import cm
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/"
Nresponse = 25000

# Physical parameters gallium
Time = 1140         # (s)
L = 0.0889             # Length of cavity (m)
H = 0.714*L      # Height of cavity (m)
g_phys = 9.81            # Gravitational acceleration (m/s^2)
rho_phys = 6.093e3      # Density (kg/m^3)
lbda_phys = 33           # Thermal conductivity (W/m K)
mu_phys = 1.81e-3        # Dynamic viscosity (Ns/m^2)
nu_phys = mu_phys / rho_phys      # Kinematic viscosity (m^2/s)
beta_phys = 1.2e-4       # Thermal expansion (1/K)
Lat_phys = 8.016e4       # Latent heat (J/kg)
cp_phys = 381           # Specific heat (J/(kgK))
alpha_phys = lbda_phys / (rho_phys * cp_phys)     # Thermal diffusivity (m^2/s)
Tm_phys = 302.78          # Melting point (K)
g_vec_phys = g_phys * np.array([0, -1])

# Setup parameters
T0_phys = 301.3
TH_phys = 311
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
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
cs = 1 / np.sqrt(3)

# Dimensionless parameters
Pr = nu_phys / alpha_phys
Ra = beta_phys * (TH_phys - T0_phys) * g_phys * H**3 / (nu_phys * alpha_phys)

# Simulation parameters
tau = 0.505
tau_inv = 1/tau
Nx = 80
Ny = 3 #np.int(0.714*Nx)
rho = 1
nu = cs**2 * (tau - 1/2)

# Calculate dependent parameters
alpha = nu / Pr

# Calculate conversion coefficients
dx = L / Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_phys
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
B = np.ones(dim)
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

g_vec = g_vec_phys / Cg

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

print("Pr =", Pr)
print("Ra =", Ra)
print("dx =", dx, "dt =", dt)
print("Nodes:", Nx, "x", Ny)
print(f"{Nt} steps")
if alpha > 1/6:
    print(f"Warning alpha = {np.round(alpha, 2)}. Can cause stability or convergence issues.")


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


def initialize(g):
    rho = np.ones((Nx, Ny))
    vel = np.zeros((Nx, Ny, 2))

    f = np.zeros((Nx, Ny, 9))
    f_new = f_eq(rho, vel, f)
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


@njit
def f_eq(rho, vel, feq):
    for j in range(Ny):
        for i in range(Nx):
            for k in range(9):
                uc = e[k, 0] * vel[i, j, 0] + e[k, 1] * vel[i, j, 1]
                feq[i, j, k] = w[k] * rho[i, j] * (1.0 + 3.0 * uc + 4.5 * uc*uc -
                                                   1.5 * (vel[i, j, 0]*vel[i, j, 0] + vel[i, j, 1]*vel[i, j, 1]))

    return feq


@njit
def collision_speedup(Omegaf, fL, f, Si):
    Bi = np.zeros((Nx, Ny, 9))
    B = (1 - fL) * (tau - 1/2) / (fL + tau - 1/2)
    for k in range(9):
        Bi[:, :, k] = B

    return B, f - (1 - Bi) * Omegaf + Bi * (f[:, :, opp] - f) + Si * (1 - Bi)


def collision(r, u, f_old, Si, fL):
    Omegaf = np.einsum('ij,klj->kli', MSM, f_old - f_eq(r, u, f_old))
    return collision_speedup(Omegaf, fL, f_old, Si)


@njit
def forcing(vel, g, Si, F, T):
    for j in range(Ny):
        for i in range(Nx):
            ip = i + 1
            jp = j + 1
            F[i, j, 0] = - T[ip, jp] * g[0]
            F[i, j, 1] = - T[ip, jp] * g[1]
            for k in range(9):
                eF = F[i, j, 0]*e[k, 0]+F[i, j, 1]*e[k, 1]
                Si[i, j, k] = (1 - 1/(2*tau)) * w[k] * (3 * eF + 9 * (vel[i, j, 0]*e[k, 0]+vel[i, j, 1]*e[k, 1]) * eF - \
                                                        3 * (vel[i, j, 0]*F[i, j, 0]+vel[i, j, 1]*F[i, j, 1]))

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

        # Ghost nodes
        T_new[1:-1, 0] = 21/23 * T_new[1:-1, 1] + 3/23 * T_new[1:-1, 2] - 1/23 * T_new[1:-1, 3]         # Neumann extrapolation on lower boundary
        T_new[1:-1, -1] = 21/23 * T_new[1:-1, -2] + 3/23 * T_new[1:-1, -3] - 1/23 * T_new[1:-1, -4]     # Neumann extrapolation on upper boundary
        T_new[0, :] = 16/5 * T_dim_H - 3 * T_new[1, :] + T_new[2, :] - 1/5 * T_new[3, :]               # Dirichlet extrapolation on left boundary
        # T_new[-1, :] = 21/23 * T_new[-2, :] + 3/23 * T_new[-3, :] - 1/23 * T_new[-4, :]               # Neumann extrapolation on right boundary
        T_new[-1, :] = 16/5 * T_dim_C - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]           # Dirichlet extrapolation on right boundary

        if np.any(np.abs(f_l - f_l_iter)) < 1e-5:
            break
        else:
            T_iter = T_new.copy()
            h_iter = h_new.copy()
            c_app_iter = c_app.copy()
            f_l_iter = f_l.copy()

        n_iter += 1

        if n_iter == 10000:
            print("No convergence in T after", n_iter, "iterations.")
            print("Timestep", t)

    return T_new, h_new, c_app, f_l


@njit
def moment_update(rho, vel, f_new, F, B):
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            if B[i, j] == 1:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j] + F[i, j, 0] / 2
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j] + F[i, j, 1] / 2

    return rho, vel, f_new


def solve(h, c_app, fL, B):
    vel, rho, f_str, f_old, Si, F, T = initialize(g_vec)
    xi = 0.13870334
    X_th = []
    X_sim = []
    t_phys = []

    # Nt = 10
    for t in range(Nt):
        rho, vel, f_old = moment_update(rho, vel, f_str, F, B)
        T, h, c_app, fL = temperature(T, h, c_app, fL, 0*vel[:, :, 0], 0*vel[:, :, 1], rho, TC, TH, t)
        Si, F = forcing(vel, g_vec, Si, F, T)
        B, f_col = collision(rho, vel, f_old, Si, fL)
        f_str = streaming(rho, f_old, f_col)

        if t % 2500 == 1:
            print(t)
            if t == 1:
                begin = time.time()
            if t == 5001:
                end = time.time()
                runtime = (end - begin) * Nt / 5000
                print("Estimated runtime:", np.round(runtime/60), "minutes.")

        if (t % 10000 == 0):
            temp = t * Time / Nt
            t_phys.append(temp)

            Xt = 2 * xi * np.sqrt(alpha_phys * temp)
            X_th.append(Xt)

            half_f_l = np.abs(fL[:, Ny // 2] - 0.5)
            idx = half_f_l.argmin()
            Xt_sim = (idx + 0.5) / Nx * L
            X_sim.append(Xt_sim)

        if t % Nresponse == 0 and t != 0:
            T_phys = T / beta_phys + T0_phys
            TH_phys = TH / beta_phys + T0_phys
            ux_phys = vel[:, :, 0] * Cu
            uy_phys = vel[:, :, 1] * Cu

            # # Liquid fraction
            # plt.figure()
            # plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'Gallium \n $f_l$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
            # plt.colorbar()
            # plt.savefig(path_name + f"heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
            #
            # # Velocities
            # plt.figure()
            # plt.clf()
            # plt.imshow(np.flip(uy_phys, axis=1).T, cmap=cm.Blues)
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'Gallium \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
            # plt.colorbar()
            # plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
            #
            # plt.figure()
            # plt.clf()
            # plt.imshow(ux_phys.T, cmap=cm.Blues, origin='lower')
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'Gallium \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
            # plt.colorbar()
            # plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
            #
            # ## Temperature heatmap
            # plt.figure()
            # plt.clf()
            # plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cm.Blues)
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'Gallium \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
            # plt.colorbar()
            # plt.savefig(path_name + f"heatmap_T_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
            #
            # plt.figure()
            # plt.clf()
            # plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'$\\rho$ in cavity with left wall at $T={TH}K$')
            # plt.colorbar()
            # plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_test1.png")
            #
            # # Vector plot
            # plt.figure()
            # plt.quiver(ux_phys.T, uy_phys.T)
            # plt.xlabel('$x$ (# lattice nodes)')
            # plt.ylabel('$y$ (# lattice nodes)')
            # plt.title(f'Gallium \n $u$ in pipe with left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
            # # plt.legend('Velocity vector')
            # plt.savefig(path_name + f"arrowplot_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
            #
            # plt.close('all')

    # Make arrays from lists
    t_phys = np.array(t_phys)
    X_th = np.array(X_th)
    X_sim = np.array(X_sim)
    plt.figure()
    plt.plot(X_th, t_phys)
    plt.plot(X_sim, t_phys)
    plt.xlabel('$x$ (m)')
    plt.ylabel('$t$ (s)')
    plt.title(f'Gallium \n Position of melting front, left wall at $T={TH}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.savefig(path_name + f"x_pos_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")

    # Liquid fraction
    plt.figure()
    plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $f_l$, left wall at $T={TH}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")

    return vel


start = time.time()

u = solve(h, c_app, fL, B)

stop = time.time()
print(stop-start)

# Stefan problem
min_3 = np.array([[0.011058988764044946, 0.0891256345177665],
                  [0.011058988764044946, 0.08596675126903555],
                  [0.010999531835205996, 0.08055152284263961],
                  [0.010404962546816481, 0.0735568527918782],
                  [0.00986985018726592, 0.06498274111675129],
                  [0.009572565543071164, 0.05663426395939088],
                  [0.009453651685393262, 0.048511421319796966],
                  [0.009275280898876406, 0.04050139593908632],
                  [0.009215823970037455, 0.032717005076142155],
                  [0.008799625468164795, 0.024142893401015247],
                  [0.008442883895131089, 0.01624568527918782],
                  [0.008205056179775282, 0.008912563451776656],
                  [0.008086142322097381, 0.004287055837563461]])
