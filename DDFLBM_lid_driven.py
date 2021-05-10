import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streaming as stream
import streaming_temp as str_temp
import steaming_lid_driven as str_lid
import sys
import time
from matplotlib import cm
from numba import njit, jit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
folder_nr = 'lid_driven_cavity/test2'

# Define constants
# # Physical parameters water
# L = 0.1             # Length of cavity (m)
# H = L               # Height of cavity (m)
# g = 9.81            # Gravitational acceleration (m/s^2)
# Time = 100         # (s)
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
Time = 40 #1080         # (s)
L = 0.1 #0.06 #0.0889             # Length of cavity (m)
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
T_H = 311           # Hot wall temperature (K)
T_C = 301 #301.3         # Hot wall temperature (K)
epsilon = 0.05 * (T_H - T_C)  # Width mushy zone (K)
umax = np.sqrt(g * beta * (T_H - T0) * H)           # Maximal velocity

# Dimensionless numbers
Re = umax * H / nu                                  # Reynolds number
Ra = beta * (T_H - T0) * g * H**3 / (nu * alpha)    # Rayleigh number
print('Ra', Ra)
Pr = nu / alpha                                      # Prandtl number

# Choose simulation parameters
Lambda = 1/4        # Magic parameter
tau_plus = 0.52      # Even relaxation time
rho0_sim = 1        # Starting simulation density
Nx = 20             # Nodes in y-direction
Ny = Nx #np.int(0.714*Nx)

dx_sim = 1          # simulation length
dt_sim = 1          # simulation time
c_s = 1 / np.sqrt(3)              # Simulation speed of sound
nu_sim = c_s**2 * (tau_plus - 1/2)                    # Simulation viscosity
print('nu_sim', nu_sim)

# Determine dependent parameters
umax_sim = Re * nu_sim / Ny                             # Maximal simulation density
print('umax_sim', umax_sim)
tau_minus = Lambda / (tau_plus - 1/2) + 1/2
print('tau_plus', tau_plus)
print('tau_minus', tau_minus)
tau_g_minus = nu_sim / (c_s**2 * Pr) + 1/2
# tau_g_minus = 0.7
tau_g_plus = Lambda / (tau_g_minus - 1/2) + 1/2
print('tau_g_plus', tau_g_plus)
print('tau_g_minus', tau_g_minus)
alpha_sim = (tau_g_minus - 1/2) * c_s**2

# if alpha_sim > 1/6:
#     print(f"alpha too large ({alpha_sim}), unstable temperature")

# Calculate conversion parameters
dx = L / Nx                                             # Distance
dt = (c_s**2) * (tau_plus - 1/2) * (dx**2 / nu)       # Time
print(dx, dt)
Cu = dx / dt                                            # Velocity
Cg = dx / dt**2                                         # Acceleration
Crho = rho0 / rho0_sim                                  # Density
Ch = dx**2 / dt**2                                      # Specific enthalpy

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
w_temp = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
q = 9                               # number of directions velocity
q_temp = 5                          # Number of directions temperature

# Grid and time steps
# Ny = np.int(H/dx)                                 # Number of lattice nodes in x-direction
Nt = np.int(Time / dt)                  # Number of time steps
print('Nt', Nt)

# Forces
g_sim = g / Cg * np.array([0, -1]) #np.array([-1, 0])   # Simulation acceleration vector

# Initial conditions
dim = (Nx, Ny)
ux = np.zeros(dim)                 # Simulation velocity in x direction
uy = np.zeros(dim)                 # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones(dim)  # Simulation density
T_dim = np.zeros(dim)     # Dimensionless simulation temperature
f_l_ts = np.ones(dim)                # Liquid fraction
h = (c_p * T0) * np.ones(dim)        # Enthalpy
c_app = c_p * np.ones(dim)

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
s7 = 1 / tau_plus
s4 = 8 * ((2 - s7) / (8 - s7))
s5 = 0
s6 = s4
s8 = s7
S = np.diag(np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8]))

# Temperature BCS
T_dim_H = np.zeros(Ny) * beta * (T_H - T0)
T_dim_C = np.zeros(Ny) * beta * (T_C - T0)

uxw = 0.00833 * 5
uyw = 0

Re2 = uxw * Nx / nu_sim
print("Re", Re2)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


def streaming(N_x, N_y, fp, fm, fs, rho_, w, cs, c, ux_w, uy_w):
    fp = np.zeros((Nx, Ny, q))
    fm = np.zeros((Nx, Ny, q))
    fp, fm = str_lid.fluid(N_x, N_y, fp, fm, fs)
    fp, fm = str_lid.left_wall(N_y, fp, fm, fs)
    fp, fm = str_lid.right_wall(N_x, N_y, fp, fm, fs)
    fp, fm = str_lid.lower_wall(N_x, fp, fm, fs)
    fp, fm = str_lid.upper_wall(N_x, N_y, fp, fm, fs, rho_, w, cs, c, ux_w, uy_w)
    fp, fm = str_lid.lower_left_corner(fp, fm, fs, w)
    fp, fm = str_lid.lower_right_corner(N_x, fp, fm, fs, w)
    fp, fm = str_lid.upper_left_corner(N_y, fp, fm, fs, rho_, w, cs, c, ux_w, uy_w)
    fp, fm = str_lid.upper_right_corner(N_x, N_y, fp, fm, fs, rho_, w, cs, c, ux_w, uy_w)

    return fp, fm


# @njit
def f_equilibrium(w_i, rho_, u_x, u_y, N_x, N_y, ci, cs, q):
    f_eq_plus = np.zeros((N_x, N_y, q))                                   # Initialize even and odd parts of f_eq
    f_eq_minus = np.zeros((N_x, N_y, q))
    ####
    f_eq = np.zeros((N_x, N_y, q))

    u_dot_u = u_x**2 + u_y**2                                             # Inner product of u with itself

    for i in range(q):                                                  # Loop over all directions of Q
        # if i == 0:                                                      # If-statement for symmetry arguments
        #     u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]                   # Inner product of u with c_i
        #     f_eq_plus[:, :, i] = w_i[i] * rho_ * (1 + (u_dot_c[:, :] / c_s**2) + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))
        # elif i in [1, 2, 5, 6]:
        #     u_dot_c = ux * c_i[i, 0] + uy * c_i[i, 1]
        #     f_eq_plus[:, :, i] = w_i[i] * rho_ * (1 + (u_dot_c[:, :]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2)))      # Even part of f_eq
        #     f_eq_minus[:, :, i] = w_i[i] * rho_ * (u_dot_c[:, :] / c_s**2)                                               # Odd part of f_eq
        # else:
        #     f_eq_plus[:, :, i] = f_eq_plus[:, :, c_opp[i]]
        #     f_eq_minus[:, :, i] = -f_eq_minus[:, :, c_opp[i]]

        u_dot_c = u_x * ci[i, 0] + u_y * ci[i, 1]
        # print(i)
        # easy_view("u_x", u_x)
        # easy_view("u_y", u_y)
        # easy_view("u_dot_c", u_dot_c)

        f_eq[:, :, i] = w_i[i] * rho_[:, :] * (1 + 3 * u_dot_c[:, :] + 4.5 * u_dot_c[:, :]**2 - 1.5 * u_dot_u[:, :])
        # f_eq[:, :, i] = w_i[i] * rho_[:, :] * (1 + u_dot_c[:, :] / cs**2 + u_dot_c[:, :]**2 / (2 * cs**4) - u_dot_u[:, :] / (2 * cs**2))


    f_eq_plus[:, :, 0] = f_eq[:, :, 0]
    f_eq_plus[:, :, 1] = (f_eq[:, :, 1] + f_eq[:, :, 3]) / 2
    f_eq_plus[:, :, 2] = (f_eq[:, :, 2] + f_eq[:, :, 4]) / 2
    f_eq_plus[:, :, 3] = (f_eq[:, :, 3] + f_eq[:, :, 1]) / 2
    f_eq_plus[:, :, 4] = (f_eq[:, :, 4] + f_eq[:, :, 2]) / 2
    f_eq_plus[:, :, 5] = (f_eq[:, :, 5] + f_eq[:, :, 7]) / 2
    f_eq_plus[:, :, 6] = (f_eq[:, :, 6] + f_eq[:, :, 8]) / 2
    f_eq_plus[:, :, 7] = (f_eq[:, :, 7] + f_eq[:, :, 5]) / 2
    f_eq_plus[:, :, 8] = (f_eq[:, :, 8] + f_eq[:, :, 6]) / 2

    f_eq_minus[:, :, 0] = 0
    f_eq_minus[:, :, 1] = (f_eq[:, :, 1] - f_eq[:, :, 3]) / 2
    f_eq_minus[:, :, 2] = (f_eq[:, :, 2] - f_eq[:, :, 4]) / 2
    f_eq_minus[:, :, 3] = (f_eq[:, :, 3] - f_eq[:, :, 1]) / 2
    f_eq_minus[:, :, 4] = (f_eq[:, :, 4] - f_eq[:, :, 2]) / 2
    f_eq_minus[:, :, 5] = (f_eq[:, :, 5] - f_eq[:, :, 7]) / 2
    f_eq_minus[:, :, 6] = (f_eq[:, :, 6] - f_eq[:, :, 8]) / 2
    f_eq_minus[:, :, 7] = (f_eq[:, :, 7] - f_eq[:, :, 5]) / 2
    f_eq_minus[:, :, 8] = (f_eq[:, :, 8] - f_eq[:, :, 6]) / 2

    return f_eq_plus, f_eq_minus


f_plus, f_minus = f_equilibrium(w_i, rho_sim, ux, uy, Nx, Ny, c_i, c_s, q)          # Initialize distributions
f_i = f_plus + f_minus

start = time.time()

X_th = []
X_sim = []
t_phys = []

Nt2 = 3
# tau_minus = tau_plus
for t in range(Nt):
    stri = f"{t}"

    ### Moment update
    rho_sim = np.sum(f_plus, axis=2)                                        # Calculate density (even parts due to symmetry)
    f = f_plus + f_minus

    B = (1 - f_l_ts) * (tau_plus - 1/2) / (f_l_ts + tau_plus - 1/2)               # Viscosity-dependent solid fraction

    # easy_view("ux", np.flip(ux, axis=1))
    ux = (f[:, :, 1] + f[:, :, 5] + f[:, :, 8] - (f[:, :, 3] + f[:, :, 6] + f[:, :, 7])) / rho_sim
    uy = (f[:, :, 2] + f[:, :, 5] + f[:, :, 6] - (f[:, :, 4] + f[:, :, 7] + f[:, :, 8])) / rho_sim

    ### Equilibrium
    f_eq_plus, f_eq_minus = f_equilibrium(w_i, rho_sim, ux, uy, Nx, Ny, c_i, c_s, q)                              # Calculate new equilibrium distribution
    f_eq = f_eq_plus + f_eq_minus

    # ### Collision
    # Mf = np.einsum('ij,klj->kli', M, f_i - f_eq)
    # SMf = np.einsum('ij,klj->kli', S, Mf)
    # MSMf = np.einsum('ij,klj->kli', M_inv, SMf)
    #
    # f_star = f_i - MSMf

    f_star = f_i - 1/tau_plus * (f_plus - f_eq_plus) - 1/tau_minus * (f_minus - f_eq_minus)

    ### Streaming
    f_plus, f_minus = streaming(Nx, Ny, f_plus, f_minus, f_star, rho_sim, w_i, c_s, c_i, uxw, uyw)
    f_i = f_plus + f_minus

    if t == 1150 and t != 0:
        T = T_dim / beta + T0
        x = np.linspace(L/Nx/2, L-L/Nx/2, len(T[5, :]))

        # List of points in x axis
        XPoints = []

        # List of points in y axis
        YPoints = []

        # X and Y points are from -6 to +6 varying in steps of 2
        vals = np.linspace(0, 0.1, Nx)
        for val in vals:
            XPoints.append(val)
            YPoints.append(val)

        # Provide a title for the contour plot
        plt.figure()
        plt.title('Streamplot velocity')
        plt.xlabel('x')
        plt.ylabel('y')
        v = ux**2 + uy**2
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)
        plt.streamplot(X, Y, ux.T, uy.T)
        plt.savefig(f"Figures/{folder_nr}/contour_plot_u_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test10.png")

        # Velocities
        plt.figure()
        plt.clf()
        plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Lid-driven cavity, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/{folder_nr}/heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}_test10.png")

        plt.figure()
        plt.clf()
        plt.imshow(ux.T, cmap=cm.Blues, origin='lower')
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Lid-driven cavity, $t={np.round(t/Nt*Time, decimals=2)}s$')
        plt.colorbar()
        plt.savefig(f"Figures/{folder_nr}/heatmap_ux_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test10.png")

        # Vector plot
        plt.figure()
        plt.quiver(Cu*ux.T, Cu*uy.T)
        plt.xlabel('$x$ (# lattice nodes)')
        plt.ylabel('$y$ (# lattice nodes)')
        plt.title(f'Lid-driven cavity, $t={np.round(t/Nt*Time, decimals=2)}s$')
        # plt.legend('Velocity vector')
        plt.savefig(f"Figures/{folder_nr}/arrowplot_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test10.png")

        plt.close('all')


T = T_dim / beta + T0
x = np.linspace(L/Nx/2, L-L/Nx/2, len(T[5, :]))

# List of points in x axis
XPoints = []

# List of points in y axis
YPoints = []

# X and Y points are from -6 to +6 varying in steps of 2
vals = np.linspace(0, 0.1, Nx)
for val in vals:
    XPoints.append(val)
    YPoints.append(val)

ux_aslan = np.array([[0, -0.0034965034965035446],
                    [-0.15774647887323945, 0.04265734265734267],
                    [-0.2676056338028169, 0.08671328671328671],
                    [-0.3549295774647887, 0.1286713286713288],
                    [-0.3830985915492957, 0.16643356643356655],
                    [-0.3464788732394366, 0.21678321678321688],
                    [-0.2788732394366197, 0.2734265734265735],
                    [-0.2084507042253521, 0.34055944055944065],
                    [-0.14366197183098584, 0.4118881118881119],
                    [-0.07605633802816902, 0.48531468531468536],
                    [0, 0.5566433566433566],
                    [0.07887323943661972, 0.6300699300699302],
                    [0.13239436619718314, 0.6846153846153846],
                    [0.21408450704225357, 0.755944055944056],
                    [0.2985915492957747, 0.8188811188811189],
                    [0.3661971830985915, 0.8755244755244757],
                    [0.41408450704225364, 0.9300699300699302],
                    [0.5042253521126763, 0.9594405594405595],
                    [0.5943661971830987, 0.96993006993007],
                    [0.7070422535211267, 0.9783216783216784],
                    [0.8140845070422535, 0.9888111888111889],
                    [0.9352112676056339, 0.9972027972027973]])
y = np.linspace(0, Nx-1, Nx)

plt.figure()
plt.title('$u_x$ at $x=L/2$')
plt.xlabel('$u_x/u_0$')
plt.ylabel('$y/H$')
plt.plot(ux[np.int(Nx/2), :]/uxw, y/Nx)
plt.plot(ux_aslan[:, 0], ux_aslan[:, 1])
plt.savefig(f"Figures/lid_driven_cavity/test4/u_compare_t={np.round(t/Nt*Time, decimals=3)}_test1.png")

# # Provide a title for the contour plot
# plt.figure()
# plt.title('Contour plot')
# plt.xlabel('x')
# plt.ylabel('y')
# v = (ux**2 + uy**2) / uxw
# contours = plt.contour(XPoints, YPoints, v.T)
# plt.clabel(contours, inline=1, fontsize=10)
# plt.savefig(f"Figures/{folder_nr}/contour_plot_u_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test10.png")
#
# plt.figure()
# contours = plt.contour(XPoints, YPoints, T[:, :].T)
# plt.clabel(contours, inline=1, fontsize=10)
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/contour_plot_T_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test5.png")
#
# # ### Moment update
# # rho_sim = np.sum(f_plus, axis=2)                                        # Calculate density (even parts due to symmetry)
# #
# # B = (1 - f_l_ts) * (tau_plus - 1/2) / (f_l_ts + tau_plus - 1/2)               # Viscosity-dependent solid fraction
# # ux = 1 * (np.sum(f_minus[:, :] * c_i[:, 0], axis=2) / rho_sim)    # Calculate x velocity (odd parts due to symmetry)
# # uy = 1 * (np.sum(f_minus[:, :] * c_i[:, 1], axis=2) / rho_sim)    # Calculate y velocity (odd parts due to symmetry)
# #
# # ux[np.round(B) == 1] = 0                                                # Force velocity in solid to zero
# # uy[B == 1] = 0
#
# # # Liquid fraction
# # plt.figure()
# # plt.imshow(f_l_ts.T, cmap=cm.autumn, origin='lower', aspect=1.0)
# # plt.xlabel('$x$ (# lattice nodes)')
# # plt.ylabel('$y$ (# lattice nodes)')
# # plt.title(f'Gallium \n $f_l$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# # plt.colorbar()
# # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_fl_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}_test.png")
#
# # Velocities
# plt.figure()
# plt.clf()
# plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'Gallium \n $u_y$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_uy_t={np.round(t/Nt*Time, decimals=2)}_N{Ny}_test.png")
#
# # plt.figure()
# # plt.clf()
# # plt.imshow(ux.T, cmap=cm.Blues, origin='lower')
# # plt.xlabel('$x$ (# lattice nodes)')
# # plt.ylabel('$y$ (# lattice nodes)')
# # plt.title(f'Gallium \n $u_x$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# # plt.colorbar()
# # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_ux_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")
#
# ## Temperature heatmap
# plt.figure()
# plt.clf()
# plt.imshow(np.flip(T[:, :].T, axis=0), cmap=cm.Blues)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'Gallium \n $T$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.colorbar()
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_T_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")
#
# # ## Temperature lineplot
# # plt.figure()
# # plt.plot(T[:, 5], x)
# # plt.xlabel('$x$ (m)')
# # plt.ylabel('$T$ (K)')
# # plt.title(f'Gallium \n $T$, left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/lineplot_T_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")
# #
# # plt.figure()
# # plt.clf()
# # plt.imshow(np.flip(rho_sim, axis=1).T, cmap=cm.Blues)
# # plt.xlabel('$x$ (# lattice nodes)')
# # plt.ylabel('$y$ (# lattice nodes)')
# # plt.title(f'$\\rho$ in cavity with left wall at $T={T_H}K$')
# # plt.colorbar()
# # plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/heatmap_rho_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")
#
# # Vector plot
# plt.figure()
# plt.quiver(Cu*ux.T, Cu*uy.T)
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title(f'Gallium \n $u$ in pipe with left wall at $T={T_H}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# # plt.legend('Velocity vector')
# plt.savefig(f"Figures/hsource_trt-fsm_sq_cav/{folder_nr}/arrowplot_t={np.round(t/Nt*Time, decimals=3)}_N{Ny}_test.png")

plt.close('all')


stop = time.time()
print(stop-start)
