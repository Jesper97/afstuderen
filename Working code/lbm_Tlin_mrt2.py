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
Nresponse = 25000

# # Domain parameters
# Time = 5000
# L = 0.1
# H = L
# g_phys = 9.81
# g_vec_phys = np.array([0, -g_phys])

# Physical parameters gallium
Time = 1140         # (s)
L = 0.01             # Length of cavity (m)
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
Tm_phys = 302.8          # Melting point (K)
g_vec_phys = g_phys * np.array([0, -1])

# # Material parameters octadecane
# rho_phys = 775.5
# cp_phys = 2200
# lbda_phys = 0.157
# mu_phys = 3.26e-3
# nu_phys = 5.005e-6 / 1.157627
# alpha_phys = 8.647e-8
# beta_phys = 9.1e-4/238.786355
# Lat_phys = 241e3 * 1.01429222
# Tm_phys = 301.13

# Temperature
DT = 10.7
T0_phys = 302.67
TH_phys = 305 #T0_phys + DT
TC_phys = 301.3 #T0_phys
epsilon = 0.01 * (TH_phys - Tm_phys)

# LBM parameters
q = 9
w0 = 4/9
ws = 1/9
wd = 1/36
w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd])
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
cs = 1/np.sqrt(3)

# Dimensionless numbers
Pr = nu_phys / alpha_phys
Ra = g_phys * beta_phys * (TH_phys - TC_phys) * H**3 / (nu_phys * alpha_phys)
Ste = cp_phys * DT / Lat_phys

# Simulation parameters
l_relax = 0.1
tau = 0.55
tau_inv = 1/tau
Nx = 40
Ny = 3 #np.int(0.714*Nx)
rho0 = 1
nu = cs**2 * (tau - 1/2)

# Calculate dependent parameters
alpha = nu / Pr

dx = L / Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_phys
Cu = dx / dt
Cg = dx / dt**2
Crho = rho_phys / rho0
Ch = dx**2 / dt**2
Ccp = Ch * beta_phys
print("Ccp", Ccp)
Clbda = Crho * dx**4 / dt**3 * beta_phys

Nt = np.int(Time/dt)

# Initial conditions
dim = (Nx, Ny)
capp = cp_phys / Ccp * np.ones(dim)
lbda = lbda_phys / Clbda
fL = np.zeros(dim)
B = np.ones(dim)
h = cp_phys / Ccp * np.zeros(dim)
Lat = Lat_phys / Ch

# Mushy zone parameters
cs = cp_phys / Ccp
cl = cp_phys / Ccp
Ts = ((Tm_phys - epsilon) - T0_phys) * beta_phys
Tl = ((Tm_phys + epsilon) - T0_phys) * beta_phys
hs = cs * Ts
hl = hs + Lat + (cs + cl) / 2 * (Tl - Ts)

dT = Tl - Ts                # Parts of formulas to speed up code
dTdh = dT / (hl - hs)
cs_LatdT = cs + (Lat / dT)
dcdT = (cl - cs) / dT

g_vec = g_vec_phys / Cg

# Temperature BCS
TH = beta_phys * (TH_phys - T0_phys)
TC = beta_phys * (TC_phys - T0_phys)

Ra_LB = -g_vec[1] * (TH - TC) * Ny**3 / (nu * alpha)

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

print("Pr =", Pr)
print("Ra =", Ra)
print("Ste =", Ste)
print("dx =", dx, "dt =", dt)
print("Nodes:", Nx, "x", Ny)
print(f"{Nt} steps")
if alpha > 1/6:
    print(f"Warning alpha = {np.round(alpha, 2)}. Can cause stability or convergence issues.")

# 1D Stefan problem
xi = 0.02286148


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


def initialize(g):
    rho = rho0 * np.ones((Nx, Ny))
    vel = np.zeros((Nx, Ny, 2))

    f_new = f_eq(rho, vel)

    T = np.zeros((Nx+2, Ny+2))

    Si = np.zeros((Nx, Ny, q))
    F = - T[1:-1, 1:-1, None] * rho[:, :, None] * g

    return vel, rho, f_new, Si, F, T


@njit
def streaming(f_old):
    f_new = np.empty((Nx, Ny, 9))
    f = str.fluid(Nx, Ny, e, f_new, f_old)
    f = str.left_right_wall(Nx, Ny, f, f_old)
    f = str.top_bottom_wall(Nx, Ny, f, f_old)
    f = str.bottom_corners(Nx, Ny, f, f_old)
    f = str.top_corners(Nx, Ny, f, f_old)

    return f


@njit
def f_eq(rho, vel):
    feq = np.empty((Nx, Ny, q))
    for j in range(Ny):
        for i in range(Nx):
            for k in range(q):
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


def collision(r, u, f_old, Si):
    Omegaf = np.einsum('ij,klj->kli', MSM, f_old - f_eq(r, u) - Si/2)
    return f_old - Omegaf + Si


@njit
def forcing(vel, g, T):
    Si = np.empty((Nx, Ny, q))
    F = np.empty((Nx, Ny, q))
    for j in range(Ny):
        for i in range(Nx):
            ip = i + 1
            jp = j + 1
            F[i, j, 0] = - T[ip, jp] * g[0] * rho0
            F[i, j, 1] = - T[ip, jp] * g[1] * rho0
            for k in range(q):
                eF = F[i, j, 0]*e[k, 0]+F[i, j, 1]*e[k, 1]
                Si[i, j, k] = w[k] * (3 * eF + 9 * (vel[i, j, 0]*e[k, 0]+vel[i, j, 1]*e[k, 1]) * eF -
                                      3 * (vel[i, j, 0]*F[i, j, 0]+vel[i, j, 1]*F[i, j, 1]))

    return Si, F


@njit
def temperature(T_iter, h_old, capp_iter, fL_old, ux, uy, rho, T_dim_C, T_dim_H, t):
    T_new = np.zeros((Nx+2, Ny+2))

    def energy_eq(i, j, T, ux, uy, c_app, h, h_old):
        im = i - 1
        jm = j - 1
        ip = i + 1
        jp = j + 1
        a_app = lbda / (capp[im, jm] * rho0)
        T_new = T[i, j] * (1 - 6 * a_app) + T[i, jm] * (uy[im, jm] + 2 * a_app) + T[i, jp] * (-uy[im, jm] + 2 * alpha) + \
                T[im, jm] * (-ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + T[im, j] * (ux[im, jm] + 2 * a_app) + \
                T[im, jp] * (-ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) + T[ip, jm] * (ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + \
                T[ip, j] * (-ux[im, jm] + 2 * a_app) + T[ip, jp] * (ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) - \
                (h[im, jm] - h_old[im, jm]) / c_app[im, jm]

        return T_new

    fL_iter = -np.ones(dim)  # Initialize
    h_iter = h_old.copy()
    fL = fL_old.copy()

    n_iter = 1
    while True:
        for j in range(1, Ny+1):
            for i in range(1, Nx+1):
                T_new[i, j] = energy_eq(i, j, T_iter, ux, uy, capp_iter, h_iter, h_old)

        h_new = h_iter + l_relax * capp_iter * (T_new[1:-1, 1:-1] - T_iter[1:-1, 1:-1])

        # print(t)
        # easy_view("T1", T_new/beta_phys + T0_phys)
        # easy_view("h_new", h_new * Ch)

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
                    capp[im, jm] = cs
                    fL[im, jm] = 0
                elif T_new[i, j] > Tl:
                    capp[im, jm] = cl
                    fL[im, jm] = 1
                else:
                    capp[im, jm] = cs_LatdT + (T_new[i, j] - Ts) * dcdT
                    fL[im, jm] = (T_new[i, j] - Ts) / dT

        # Ghost nodes
        T_new[1:-1, 0] = 21/23 * T_new[1:-1, 1] + 3/23 * T_new[1:-1, 2] - 1/23 * T_new[1:-1, 3]         # Neumann extrapolation on lower boundary
        T_new[1:-1, -1] = 21/23 * T_new[1:-1, -2] + 3/23 * T_new[1:-1, -3] - 1/23 * T_new[1:-1, -4]     # Neumann extrapolation on upper boundary
        T_new[0, :] = 16/5 * T_dim_H - 3 * T_new[1, :] + T_new[2, :] - 1/5 * T_new[3, :]               # Dirichlet extrapolation on left boundary
        # T_new[-1, :] = 21/23 * T_new[-2, :] + 3/23 * T_new[-3, :] - 1/23 * T_new[-4, :]               # Neumann extrapolation on right boundary
        T_new[-1, :] = 16/5 * T_dim_C - 3 * T_new[-2, :] + T_new[-3, :] - 1/5 * T_new[-4, :]           # Dirichlet extrapolation on right boundary

        # easy_view("T2", T_new/beta_phys + T0_phys)
        # easy_view("capp", capp * Ccp)
        # easy_view("fL", fL)

        if np.any(np.abs(fL - fL_iter)) < 1e-5:
            break
        else:
            T_iter = T_new.copy()
            h_iter = h_new.copy()
            capp_iter = capp.copy()
            fL_iter = fL.copy()

        n_iter += 1

        if n_iter == 10000:
            print("No convergence in T after", n_iter, "iterations.")
            print("Timestep", t)

    return T_new


@njit
def moment_update(f_new, F, B):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            if B[i, j] != 0:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j] + F[i, j, 0] / (2 * rho0)
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j] + F[i, j, 1] / (2 * rho0)

    return rho, vel


@njit
def moment_plots(f_new, B):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            if B[i, j] != 0:
                vel[i, j, 0] = 0
                vel[i, j, 1] = 0
            else:
                vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j]
                vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j]

    return rho, vel


def solve(h, capp, fL, B):
    vel, rho, f_str, Si, F, T = initialize(g_vec)

    for t in range(Nt):
        rho, vel = moment_update(f_str, F, B)
        T = temperature(T, h, capp, fL, 0*vel[:, :, 0], 0*vel[:, :, 1], rho, TC, TH, t)
        Si, F = forcing(vel, g_vec, T)
        f_col = collision(rho, vel, f_str, Si)
        f_str = streaming(f_col)

        ### Plots
        if (t % 1000 == 0):
            temp = t * Time / Nt
            t_phys.append(temp)

            Xt = 2 * xi * np.sqrt(alpha_phys * temp)
            X_th.append(Xt)

            # easy_view(1, f_l)
            # easy_view(2, T_dim / beta + T0)
            half_f_l = np.abs(fL[:, 2] - 0.5)
            idx = half_f_l.argmin()
            Xt_sim = (idx + 0.5) / Nx * L
            X_sim.append(Xt_sim)

        if t % 2500 == 0:
            print(t)
            print(np.max(vel[:, :, 1]))

            if t == 0:
                begin = time.time()
            if t == 7500:
                end = time.time()
                runtime = (end - begin) * Nt / 7500
                mins = np.round(runtime/60, 1)
                print(f"Estimated runtime: {mins} minutes.")

        # if (t % Nresponse == 0) and (t != 0):
        #     rho, vel = moment_plots(f_str, B)
        #     T_phys = T / beta_phys + T0_phys
        #     TH_phys = TH / beta_phys + T0_phys
        #     ux_phys = vel[:, :, 0] * Cu * L / alpha_phys
        #     uy_phys = vel[:, :, 1] * Cu * L / alpha_phys
        #
        #     # Liquid fraction
        #     plt.figure()
        #     plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Air \n $f_l$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.colorbar()
        #     plt.savefig(path_name + f"heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     # Streamlines velocity
        #     uy_plot = np.rot90(uy_phys)
        #     ux_plot = ux_phys.T
        #
        #     plt.clf()
        #     plt.figure()
        #     x = np.linspace(0, 1, Nx)
        #     y = np.linspace(0, 1, Ny)
        #     u = np.linspace(0, 1, 100)
        #     g = np.meshgrid(u, u)
        #     str_pts = list(zip(*(x.flat for x in g)))
        #     plt.streamplot(x, y, ux_plot, uy_plot,
        #                    linewidth    = 1.5,
        #                    cmap         = 'RdBu_r',
        #                    arrowstyle   = '-',
        #                    start_points = str_pts,
        #                    density      = 3)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.savefig(path_name + f"streamlines_u_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #     plt.close()
        #
        #     # Contour plots
        #     X, Y = np.meshgrid(x, y)
        #     plt.figure()
        #     CS = plt.contour(X, Y, np.flip(uy_plot, axis=1))
        #     plt.clabel(CS, inline=True)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.savefig(path_name + f"contour_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     plt.figure()
        #     plt.clf()
        #     CS = plt.contour(X, Y, ux_plot)
        #     plt.clabel(CS, inline=True)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.savefig(path_name + f"contour_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     # Velocities
        #     plt.figure()
        #     plt.clf()
        #     plt.imshow(uy_plot, cmap=cm.Blues)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes) ')
        #     plt.title(f'Octadecane \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.colorbar()
        #     plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     plt.figure()
        #     plt.clf()
        #     plt.imshow(ux_plot, cmap=cm.Blues, origin='lower')
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.colorbar()
        #     plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     # Temperature heatmap
        #     cmap = cm.get_cmap('PiYG', 11)
        #     plt.figure()
        #     plt.clf()
        #     plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cmap)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     plt.colorbar()
        #     plt.savefig(path_name + f"heatmap_T_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     plt.figure()
        #     plt.clf()
        #     plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $\\rho$, left wall at $T={TH_phys}K$')
        #     plt.colorbar()
        #     plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     # Vector plot
        #     plt.figure()
        #     plt.quiver(ux_plot, np.flip(uy_plot, axis=1))
        #     plt.xlabel('$x$ (# lattice nodes)')
        #     plt.ylabel('$y$ (# lattice nodes)')
        #     plt.title(f'Octadecane \n $u$ in pipe with left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
        #     # plt.legend('Velocity vector')
        #     plt.savefig(path_name + f"arrowplot_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
        #
        #     plt.close('all')

    return vel


start = time.time()

#####
X_th = []
X_sim = []
t_phys = []

u = solve(h, capp, fL, B)

#####
# Make arrays from lists
t_phys = np.array(t_phys)
X_th = np.array(X_th)
X_sim = np.array(X_sim)
plt.figure()
plt.plot(X_th, t_phys)
plt.plot(X_sim, t_phys)
plt.xlabel('$x$ (m)')
plt.ylabel('$t$ (s)')
plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={Time}s$')
plt.savefig(path_name + f"x_pos_t={Time}_N{Nx}_test1.png")

# Liquid fraction
plt.figure()
plt.imshow(fL.T, cmap=cm.autumn, origin='lower', aspect=1.0)
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'Gallium \n $f_l$, left wall at $T={TH_phys}K$, $t={Time}s$')
plt.colorbar()
plt.savefig(path_name + f"heatmap_fl_t={Time}_N{Nx}_test1.png")

stop = time.time()
print(stop-start)
