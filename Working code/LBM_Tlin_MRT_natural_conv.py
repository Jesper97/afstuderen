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

# Domain parameters
Time = 150
L = 0.1
H = L
g_phys = 9.81
g_vec_phys = np.array([0, -g_phys])

# Material parameters air
rho_phys = 0.9458
cp_phys = 1009
lbda_phys = 0.03095
alpha_phys = 3.243e-5 * 0.7/0.71
nu_phys = 2.2701e-5
beta_phys = 0.0034

# Temperature
DT = 22.072118186724223 / 1 * 0.7/0.71
T0_phys = 300
TH_phys = T0_phys + DT/2
TC_phys = T0_phys - DT/2

# Dimensionless numbers
Pr = nu_phys / alpha_phys
Ra = g_phys * beta_phys * (TH_phys - TC_phys) * H**3 / (nu_phys * alpha_phys)

# LBM parameters
q = 9
w0 = 4/9
ws = 1/9
wd = 1/36
w = np.array([w0, ws, ws, ws, ws, wd, wd, wd, wd])
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cs = 1/np.sqrt(3)

# Simulation parameters
tau = 0.55
tau_inv = 1/tau
Nx = 100
Ny = Nx
rho0 = 1

umax = 0.1
bc_value = np.array([[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

# Calculate dependent parameters
nu = cs**2 * (tau - 1/2)
alpha = nu / Pr

dx = L/Nx
dt = cs**2 * (tau - 1/2) * dx**2 / nu_phys
Cu = dx / dt
Cg = dx / dt**2
Crho = rho_phys / rho0
Ch = dx**2 / dt**2
Ccp = Ch * beta_phys
Clbda = Crho * dx**4 / dt**3 * beta_phys

Nt = np.int(Time/dt)

# Initial conditions
dim = (Nx, Ny)
cp = cp_phys / Ccp * np.ones(dim)
lbda = lbda_phys / Clbda

g_vec = g_vec_phys / Cg

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
print("Ra_LB =", Ra_LB)
print("dx =", dx, "dt =", dt)
print("Nodes:", Nx, "x", Ny)
print(f"{Nt} steps")
if alpha > 1/6:
    print(f"Warning alpha = {np.round(alpha, 2)}. Can cause stability or convergence issues.")

path_name = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra106/figures/"
suffix = f"_naturalconv_Ra{np.int(Ra)}_tau={tau}_N={Nx}x{Ny}.png"
csv_path = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra106/sim_data/"
csv_file = f"_naturalconv_Ra{np.int(Ra)}_tau={tau}_N={Nx}x{Ny}.csv"


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
def streaming(f_new, f_old):
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


def collision(r, u, f_old, Si):
    Omegaf = np.einsum('ij,klj->kli', MSM, f_old - f_eq(r, u) - Si/2)
    # Omegaf = tau_inv * (f_old - f_eq(r, u))

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
def temperature(T_iter, cp_iter, ux, uy, rho, T_dim_C, T_dim_H, t):
    T_new = np.empty((Nx+2, Ny+2))

    def energy_eq(i, j, T, ux, uy, rho, cp):
        im = i - 1
        jm = j - 1
        ip = i + 1
        jp = j + 1
        a_app = alpha #lbda / (cp[im, jm] * rho[im, jm])
        T_new = T[i, j] * (1 - 6 * a_app) + T[i, jm] * (uy[im, jm] + 2 * a_app) + T[i, jp] * (-uy[im, jm] + 2 * alpha) + \
                T[im, jm] * (-ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + T[im, j] * (ux[im, jm] + 2 * a_app) + \
                T[im, jp] * (-ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2) + T[ip, jm] * (ux[im, jm] / 4 - uy[im, jm] / 4 - a_app / 2) + \
                T[ip, j] * (-ux[im, jm] + 2 * a_app) + T[ip, jp] * (ux[im, jm] / 4 + uy[im, jm] / 4 - a_app / 2)

        return T_new

    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            T_new[i, j] = energy_eq(i, j, T_iter, ux, uy, rho, cp_iter)

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
def moment_update(f_new, F):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j] + F[i, j, 0] / (2 * rho0)
            vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j] + F[i, j, 1] / (2 * rho0)

    return rho, vel


@njit
def moment_plots(f_new):
    rho = np.empty((Nx, Ny))
    vel = np.empty((Nx, Ny, 2))
    for j in range(Ny):
        for i in range(Nx):
            rho[i, j] = np.sum(f_new[i, j, :])
            vel[i, j, 0] = (f_new[i, j, 1] + f_new[i, j, 5] + f_new[i, j, 8] - (f_new[i, j, 3] + f_new[i, j, 6] + f_new[i, j, 7])) / rho[i, j]
            vel[i, j, 1] = (f_new[i, j, 2] + f_new[i, j, 5] + f_new[i, j, 6] - (f_new[i, j, 4] + f_new[i, j, 7] + f_new[i, j, 8])) / rho[i, j]

    return rho, vel


@njit(fastmath=True)
def nusselt(T, t):
    Nu = np.float32(-(1 / (beta_phys * DT)) * np.sum(T[1, 1:-1] - T[0, 1:-1]))
    # Nu = -(1 / (beta_phys * DT)) * npsum((-23 * T[0, 1:-1] + 21 * T[1, 1:-1] + 3 * T[2, 1:-1] - T[3, 1:-1]) / 24)
    return Nu


def solve():
    vel, rho, f_str, Si, F, T = initialize(g_vec)

    for t in range(Nt):
        rho, vel = moment_update(f_str, F)
        T = temperature(T, cp, vel[:, :, 0], vel[:, :, 1], rho, TC, TH, t)
        Si, F = forcing(vel, g_vec, T)
        f_col = collision(rho, vel, f_str, Si)
        f_str = streaming(f_empty, f_col)

        if t % 2500 == 0:
            print(t)
            # print(np.max(vel[:, :, 1]))

            if t == 0:
                begin = time.time()
            if t == 7500:
                end = time.time()
                runtime = (end - begin) * Nt / 7500
                mins = np.round(runtime/60, 1)
                print(f"Estimated runtime: {mins} minutes.")

    Nu = nusselt(T, t)

    rho, vel = moment_plots(f_str)
    T_phys = T / beta_phys + T0_phys
    TH_phys = TH / beta_phys + T0_phys
    ux_phys = vel[:, :, 0] * Cu * L / alpha_phys
    uy_phys = vel[:, :, 1] * Cu * L / alpha_phys
    z = (np.argmax(ux_phys[Nx // 2, :]) + 1/2) / Ny
    x = (np.argmax(uy_phys[:, Ny // 2]) + 1/2) / Nx

    print("u", np.max(ux_phys[Nx // 2, :]))
    print("z", z)
    print("w", np.max(uy_phys[:, Ny // 2]))
    print("x", x)
    print("Nu", Nu)

    max_u = np.array([np.max(ux_phys[Nx // 2, :])])
    z_max_u = np.array([z])
    max_w = np.array([np.max(uy_phys[:, Ny // 2])])
    x_max_w = np.array([x])

    np.savetxt(csv_path+"_Nu"+csv_file,       np.array([Nu]),   delimiter=",")
    np.savetxt(csv_path+"_max_u"+csv_file,    max_u,   delimiter=",")
    np.savetxt(csv_path+"_z_max_u"+csv_file,  z_max_u, delimiter=",")
    np.savetxt(csv_path+"_max_w"+csv_file,    max_w,   delimiter=",")
    np.savetxt(csv_path+"_x_max_w"+csv_file,  x_max_w, delimiter=",")

    np.savetxt(csv_path+"_T"+csv_file,    T_phys,   delimiter=",")
    np.savetxt(csv_path+"_u"+csv_file,    ux_phys,   delimiter=",")
    np.savetxt(csv_path+"_w"+csv_file,    uy_phys,   delimiter=",")

    # Streamlines velocity
    uy_plot = np.rot90(uy_phys)
    ux_plot = ux_phys.T

    # plt.clf()
    # plt.figure()
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    u = np.linspace(0, 1, 100)
    g = np.meshgrid(u, u)
    # str_pts = list(zip(*(x.flat for x in g)))
    # plt.streamplot(x, y, ux_plot, np.flip(uy_plot, axis=0),
    #                linewidth    = 1.5,
    #                cmap         = 'RdBu_r',
    #                arrowstyle   = '-',
    #                start_points = str_pts,
    #                density      = 1)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.savefig(path_name + f"streamlines_u_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    # plt.close()

    # Contour plots
    X, Y = np.meshgrid(x, y)
    plt.figure()
    CS = plt.contour(X, Y, np.flip(uy_plot, axis=1))
    plt.clabel(CS, inline=True)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Air \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.savefig(path_name + f"_contour_u" + suffix)

    plt.figure()
    plt.clf()
    CS = plt.contour(X, Y, ux_plot)
    plt.clabel(CS, inline=True)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Air \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.savefig(path_name + f"_contour_w" + suffix)
    #
    # # Velocities
    # plt.figure()
    # plt.clf()
    # plt.imshow(uy_plot, cmap=cm.Blues)
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes) ')
    # plt.title(f'Air \n $u_y$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(ux_plot, cmap=cm.Blues, origin='lower')
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Air \n $u_x$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # plt.colorbar()
    # plt.savefig(path_name + f"heatmap_ux_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    ## Temperature heatmap
    cmap = cm.get_cmap('PiYG', 11)
    plt.figure()
    plt.clf()
    plt.imshow(np.flip(T_phys[1:-1, 1:-1].T, axis=0), cmap=cmap)
    plt.xlabel('$x$ (# lattice nodes)')
    plt.ylabel('$y$ (# lattice nodes)')
    plt.title(f'Air \n $T$, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    plt.colorbar()
    plt.savefig(path_name + f"_contour_T" + suffix)
    #
    # # plt.figure()
    # # plt.clf()
    # # plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
    # # plt.xlabel('$x$ (# lattice nodes)')
    # # plt.ylabel('$y$ (# lattice nodes)')
    # # plt.title(f'Air \n $\\rho$, left wall at $T={TH_phys}K$')
    # # plt.colorbar()
    # # plt.savefig(path_name + f"heatmap_rho_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)
    #
    # # Vector plot
    # plt.figure()
    # plt.quiver(ux_plot, np.flip(uy_plot, axis=1))
    # plt.xlabel('$x$ (# lattice nodes)')
    # plt.ylabel('$y$ (# lattice nodes)')
    # plt.title(f'Air \n $u$ in pipe with left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
    # # plt.legend('Velocity vector')
    # plt.savefig(path_name + f"arrowplot_t={np.round(t/Nt*Time, decimals=2)}_N{Nx}" + suffix)

    plt.close('all')

    return vel


start = time.time()

f_empty = np.empty((Nx, Ny, q))

u = solve()

stop = time.time()
print(stop-start)

