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
suffix = "Ra10^5_MRT_Guo_Fuentes.png"
Nresponse = 125000

# # Physical parameters gallium
# Time = 1140         # (s)
# L = 0.0889             # Length of cavity (m)
# H = 0.714*L      # Height of cavity (m)
# g_phys = 9.81            # Gravitational acceleration (m/s^2)
# rho_phys = 6.093e3      # Density (kg/m^3)
# lbda_phys = 33           # Thermal conductivity (W/m K)
# mu_phys = 1.81e-3        # Dynamic viscosity (Ns/m^2)
# nu_phys = mu_phys / rho_phys      # Kinematic viscosity (m^2/s)
# beta_phys = 1.2e-4       # Thermal expansion (1/K)
# Lat_phys = 8.016e4       # Latent heat (J/kg)
# cp_phys = 381           # Specific heat (J/(kgK))
# alpha_phys = lbda_phys / (rho_phys * cp_phys)     # Thermal diffusivity (m^2/s)
# Tm_phys = 302.78          # Melting point (K)
# g_vec_phys = g_phys * np.array([0, -1])

# Physical parameters air
Time = 200
L = 0.0985125
H = L
g_phys = 9.81

## Pr = 0.71
rho_phys = 0.9458
lbda_phys = 0.03095
nu_phys = 2.306e-5
beta_phys = 0.0079743   #0.0034
cp_phys = 1.009e3

## Pr = 0.77
# rho_phys = 1.284
# lbda_phys = 2.624e-2
# nu_phys = 1.568e-5
# beta_phys = 0.0034
# cp_phys = 1.0049e3
alpha_phys = lbda_phys / (rho_phys * cp_phys)
print(alpha_phys)
g_vec_phys = g_phys * np.array([0, -1])

# Setup parameters
T0_phys = 302
TH_phys = 302.5
TC_phys = 301.5

umax = 0.1
bc_value = np.array([[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

# LBM parameters
q = 9
w0 = 4/9
ws = 1/9
wd = 1/36
w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd], dtype=np.float32)
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
cs = 1 / np.sqrt(3)

# Dimensionless parameters
Pr = nu_phys / alpha_phys
Ra = beta_phys * (TH_phys - TC_phys) * g_phys * H**3 / (nu_phys * alpha_phys)

# Simulation parameters
tau = 0.56
tau_inv = 1/tau
Nx = 101
Ny = Nx #np.int(0.714*Nx)
rho0 = 1
nu = cs**2 * (tau - 1/2)

# Calculate dependent parameters
alpha = nu / Pr

# Calculate conversion coefficients
dx = L / Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_phys
Cu = dx / dt
Cg = dx / dt**2
Crho = rho_phys / rho0
Ch = dx**2 / dt**2
Ccp = Ch * beta_phys
Clbda = dx**4 / dt**3 * beta_phys * Crho

# Number of steps
Nt = np.int(Time/dt)

# Initial conditions
dim = (Nx, Ny)
rho = rho0 * np.ones(dim)  # Simulation density
T = np.zeros((Nx+2, Ny+2))     # Dimensionless simulation temperature
fL = np.ones(dim)                # Liquid fraction
B = np.zeros(dim)
h = (cp_phys / Ccp) * np.zeros(dim)        # Enthalpy
c_app = cp_phys / Ccp * np.ones(dim)
lbda = lbda_phys / Clbda

g_vec = g_vec_phys / Cg

# Temperature BCS
TH = beta_phys * (TH_phys - T0_phys)
TC = beta_phys * (TC_phys - T0_phys)

Ra_LB = (TH - TC) * np.linalg.norm(g_vec) * Nx**3 / (nu * alpha)
print("Ra_LB", Ra_LB)

# Collision operators
M_rho = np.ones(q)
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

# # Original
# s0 = 0
# s1 = 1.4
# s2 = 1.4
# s3 = 0
# s7 = 1/tau
# s4 = 1.2 #8 * ((2 - s7) / (8 - s7))
# s5 = 0
# s6 = s4
# s8 = s7

# Fuentes
s0 = 0
s1 = 1.64
s2 = 1.2
s3 = 0
s7 = 1/tau
s4 = 8 * ((2 - s7) / (8 - s7))
s5 = 0
s6 = s4
s8 = s7
S = np.diag(np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8]))

MSM = np.dot(M_inv, np.dot(S, M))

# Stefan problem
xi = 0.13870334

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

    f = np.zeros((Nx, Ny, q))
    f_new = f_eq(rho, vel, f)
    f_old = f_new.copy()

    T = np.zeros((Nx+2, Ny+2))

    Si = np.zeros((Nx, Ny, q))
    F = - T[1:-1, 1:-1, None] * rho[:, :, None] * g

    return vel, rho, f_new, f_old, Si, F, T


@njit
def streaming(rho, f_new, f_old):
    f_new = np.empty((Nx, Ny, 9))
    f = str.fluid(Nx, Ny, e, f_new, f_old)
    f = str.left_right_wall(Nx, Ny, f, f_old)
    f = str.top_bottom_wall(Nx, Ny, f, f_old, w, rho, bc_value)
    f = str.bottom_corners(Nx, Ny, f, f_old)
    f = str.top_corners(Nx, Ny, f, f_old, w, rho, bc_value)

    return f


@njit
def f_eq(rho, vel, feq):
    feq = np.empty((Nx, Ny, q))
    for j in range(Ny):
        for i in range(Nx):
            for k in range(q):
                uc = e[k, 0] * vel[i, j, 0] + e[k, 1] * vel[i, j, 1]
                feq[i, j, k] = w[k] * rho[i, j] * (1.0 + 3.0 * uc + 4.5 * uc*uc -
                                                   1.5 * (vel[i, j, 0]*vel[i, j, 0] + vel[i, j, 1]*vel[i, j, 1]))
    return feq


@njit
def collision_speedup(Omegaf, f, Si):
    Bi = np.zeros((Nx, Ny, q))
    f_new = f - Omegaf + Si

    return f_new


def collision(r, u, f_old, Si):
    Omegaf = np.einsum('ij,klj->kli', MSM, f_old - f_eq(r, u, f_old) - Si/2)
    # Omegaf = tau_inv * (f_old - f_eq(r, u, f_old))
    # return collision_speedup(Omegaf, f_old, Si)
    return f_old - Omegaf + Si


@njit
def forcing(vel, g, Si, F, T):
    Si = np.empty((Nx, Ny, q))
    F = np.empty((Nx, Ny, q))
    for j in range(Ny):
        for i in range(Nx):
            ip = i + 1
            jp = j + 1
            F[i, j, 0] = - T[ip, jp] * g[0] * rho0
            F[i, j, 1] = - T[ip, jp] * g[1] * rho0
            # F[i, j, 0] = - T[ip, jp] * g[0] * rho[i, j]
            # F[i, j, 1] = - T[ip, jp] * g[1] * rho[i, j]
            for k in range(q):
                eF = F[i, j, 0]*e[k, 0]+F[i, j, 1]*e[k, 1]
                # Si[i, j, k] = (1 - 1/(2*tau)) * w[k] * (3 * eF +
                #                                         9 * (vel[i, j, 0]*e[k, 0]+vel[i, j, 1]*e[k, 1]) * eF -
                #                                         3 * (vel[i, j, 0]*F[i, j, 0]+vel[i, j, 1]*F[i, j, 1]))
                Si[i, j, k] = w[k] * (3 * eF + 9 * (vel[i, j, 0]*e[k, 0]+vel[i, j, 1]*e[k, 1]) * eF -
                                      3 * (vel[i, j, 0]*F[i, j, 0]+vel[i, j, 1]*F[i, j, 1]))
                # Si[i, j, k] = 3 * w[k] * eF
    return Si, F


@njit
def temperature(T_iter, c_app_iter, ux, uy, rho, T_dim_C, T_dim_H, t):
    T_new = np.empty((Nx+2, Ny+2))

    def energy_eq(i, j, T, ux, uy, rho, c_app):
        im = i - 1
        jm = j - 1
        ip = i + 1
        jp = j + 1
        a_app = lbda / (c_app[im, jm] * rho[im, jm])
        T_new = T[i, j] * (1 - 6 * a_app) + T[i, j-1] * (uy[im, jm] + 2 * a_app) + T[i, j+1] * (-uy[im, jm] + 2 * alpha) + \
                T[i-1, j-1] * (-ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + T[i-1, j] * (ux[im, jm] + 2 * a_app) + \
                T[i-1, j+1] * (-ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) + T[i+1, j-1] * (ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + \
                T[i+1, j] * (-ux[im, jm] + 2 * a_app) + T[i+1, j+1] * (ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2)

        return T_new

    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            T_new[i, j] = energy_eq(i, j, T_iter, ux, uy, rho, c_app_iter)

    # Ghost nodes
    T_new[1:-1, 0] = 21/23 * T_new[1:-1, 1] + 3/23 * T_new[1:-1, 2] - 1/23 * T_new[1:-1, 3]         # Neumann extrapolation on lower boundary
    T_new[1:-1, -1] = 21/23 * T_new[1:-1, -2] + 3/23 * T_new[1:-1, -3] - 1/23 * T_new[1:-1, -4]     # Neumann extrapolation on upper boundary
    T_new[0, :] = 16/5 * T_dim_H - 3 * T_new[1, :] + T_new[2, :] - 1/5 * T_new[3, :]               # Dirichlet extrapolation on left boundary
    # T_new[-1, :] = 21/23 * T_new[-2, :] + 3/23 * T_new[-3, :] - 1/23 * T_new[-4, :]               # Neumann extrapolation on right boundary
    T_new[-1, :] = 16/5 * T_dim_C - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]           # Dirichlet extrapolation on right boundary

    ####
    # T_new[0, :] = 8/3 * T_dim_H - 2 * T_new[1, :] + T_new[2, :] / 3
    # T_new[-1, :] = 8/3 * T_dim_C - 2 * T_new[-2, :] + T_new[-3, :] / 3

    return T_new


@njit
def moment_update(rho, vel, f_new, F, B):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            if B[i, j] == 1:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j] + F[i, j, 0] / (2 * rho0)
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j] + F[i, j, 1] / (2 * rho0)

    return rho, vel, f_new

@njit
def moment_plots(f_new, B):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            if B[i, j] == 1:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j]
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j]

    return rho, vel


def solve(h, c_app, fL, B):
    vel, rho, f_str, f_old, Si, F, T = initialize(g_vec)

    for t in range(Nt):
        rho, vel, f_old = moment_update(rho, vel, f_str, F, B)
        T = temperature(T, c_app, vel[:, :, 0], vel[:, :, 1], rho, TC, TH, t)
        Si, F = forcing(vel, g_vec, Si, F, T)
        f_col = collision(rho, vel, f_old, Si)
        f_str = streaming(rho, f_old, f_col)

        if t % 2500 == 0:
            print(t)
            print(np.max(np.sqrt(vel[:, :, 0]**2+vel[:, :, 1]**2)))
            if t == 0:
                begin = time.time()
            if t == 7500:
                end = time.time()
                runtime = (end - begin) * Nt / 7500
                mins = np.round(runtime/60, 1)
                print(f"Estimated runtime: {mins} minutes.")

        # if t % Nresponse == 0 and t != 0:

    rho, vel = moment_plots(f_str, B)
    T_phys = T / beta_phys + T0_phys
    TH_phys = TH / beta_phys + T0_phys
    ux_phys = vel[:, :, 0] * Cu * L / alpha_phys
    uy_phys = vel[:, :, 1] * Cu * L / alpha_phys

    print("u_LB", np.max(vel[:, :, 0]))
    print("w_LB", np.max(vel[:, :, 1]))
    print("u_LB-", np.min(vel[:, :, 0]))
    print("w_LB-", np.min(vel[:, :, 1]))
    print("u", np.max(ux_phys))
    print("w", np.max(uy_phys))


    # # Liquid fraction
    # plt.figure()
    # plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Gallium \n $f_l$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    # Streamlines velocity
    uy_plot = np.rot90(uy_phys)
    ux_plot = ux_phys.T

    plt.clf()
    plt.figure()
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    u = np.linspace(0, 1, 100)
    g = np.meshgrid(u, u)
    str_pts = list(zip(*(x.flat for x in g)))
    plt.streamplot(x, y, ux_plot, uy_plot,
                   linewidth    = 1.5,
                   cmap         = 'RdBu_r',
                   arrowstyle   = '-',
                   start_points = str_pts,
                   density      = 3)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.savefig(path_name + f"streamlines_u_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    plt.close()

    # Contour plots
    X, Y = np.meshgrid(x, y)
    plt.figure()
    CS = plt.contour(X, Y, np.flip(uy_plot, axis=1))
    plt.clabel(CS, inline=True)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.savefig(path_name + f"contour_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    plt.figure()
    plt.clf()
    CS = plt.contour(X, Y, ux_plot)
    plt.clabel(CS, inline=True)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.savefig(path_name + f"contour_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    # Velocities
    plt.figure()
    plt.clf()
    plt.imshow(uy_plot, cmap=cm.Blues)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes) ')
    plt.title(f'Gallium \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    plt.figure()
    plt.clf()
    plt.imshow(ux_plot, cmap=cm.Blues, origin='lower')
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    ## Temperature heatmap
    cmap = cm.get_cmap('PiYG', 11)
    plt.figure()
    plt.clf()
    plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cmap)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_T_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    plt.figure()
    plt.clf()
    plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'$\\rho$ in cavity with left wall at $T={TH}K$')
    plt.colorbar()
    plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    # Vector plot
    plt.figure()
    plt.quiver(ux_plot, np.flip(uy_plot, axis=1))
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Gallium \n $u$ in pipe with left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.legend('Velocity vector')
    plt.savefig(path_name + f"arrowplot_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

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
    # plt.title(f'Gallium \n Position of melting front, left wall at $T={TH}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.savefig(path_name + f"x_pos_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")
    #
    # # Liquid fraction
    # plt.figure()
    # plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Gallium \n $f_l$, left wall at $T={TH}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}_test1.png")

    return vel


start = time.time()

u = solve(h, c_app, fL, B)

stop = time.time()
print(stop-start)

# # Compare results with literature
# y_ref, u_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 2))
# x_ref, v_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 8))
#
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
# axes.plot(u[Ny // 2, :, 0] / umax, np.linspace(0, 1.0, Ny), 'b-', label='LBM')
# axes.plot(u_ref, y_ref, 'rs', label='Ghia et al. 1982')
# axes.legend()
# axes.set_xlabel(r'$u_x$')
# axes.set_ylabel(r'$y$')
# plt.tight_layout()
# plt.savefig(path_name + "ux" + suffix)
#
# plt.clf()
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
# axes.plot(np.linspace(0, 1.0, Nx), u[:, Nx // 2, 1] / umax, 'b-', label='LBM')
# axes.plot(x_ref, v_ref, 'rs', label='Ghia et al. 1982')
# axes.legend()
# axes.set_xlabel(r'$u_x$')
# axes.set_ylabel(r'$y$')
# plt.tight_layout()
# plt.savefig(path_name + "uy" + suffix)
