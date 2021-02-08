import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from numba import jit
from matplotlib import cm

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Define constants
# Physical parameters
L = 0.01            # m
H = 0.01            # m
g = 9.81            # m/s^2
T = 1               # s
nu = 10e-6          # m^2/s
rho = 1e3           # kg/m^3
dp = 0.1            # kg/(m s^2)
# dx = H / N_y      # m
Fp = dp / L         # Pressure force density (kg/(m^2 s^2))
umax = 0.1

# Dimensionless numbers
Re = umax * H / nu
Pr = 7
Ra = 10e6
Ma = 0.1

# Chose simulation parameters to determine the dependent one
tau = 0.9
w = 100

dx_sim = 1      # simulation length
dt_sim = 1      # simulation time
c_s = (1 / np.sqrt(3)) * (dx_sim / dt_sim)  # speed of sound
nu_sim = c_s**2 * (tau - 1/2)

def check_stability(umax_sim, tau):
    if umax_sim > 0.1:
        print('Simulation velocity is', umax_sim, 'and might give unstable simulations.')
    if tau < 1/2:
        print('Relaxation time is', tau, 'and might give unstable simulations.')

# Determine dependent parameter
umax_sim = Re * nu_sim / w
check_stability(umax_sim, tau)

# Calculate conversion parameters
dx = H / w
dt = c_s**2 * (tau - 1/2) * dx**2 / nu
Cu = dx / dt
Cg = dx / dt**2

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
q = 9                               # number of directions

# Simulation parameters
alpha_sim = Pr / nu_sim

# Grid and time steps
l = w                               # lattice nodes in the x-direction
Nt = T / dt       # time steps

# Forces
Fp_sim = (Fp * w_i / c_s**2) * np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
g_sim = g / Cg
gi_sim = (g_sim * w_i / c_s**2) * np.array([0, 0, -1, 0, 1, -1, -1, 1, 1])

# Initial conditions
ux = np.zeros((l, N_y))       # Simulation velocity in x direction
uy = np.zeros((l, N_y))       # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones((l, N_y))   # Simulation density
T_dim = np.zeros((l, N_y))    # Dimensionless simulation temperature

# Temperature BCS
beta_p = 210e-6
T0 = 293
T_up_p = 289
T_down_p = 297
T_BC_upper = np.ones(l) * beta_p * (T_up_p - T0)
T_BC_lower = np.ones(l) * beta_p * (T_down_p - T0)

for node in range(len(T_BC_upper)):
    if node <= len(T_BC_upper)/2:
        T_BC_lower[node] *= -1


# Create boundary mask
def pipe_boundary(N_x, N_y):

    grid = np.zeros((N_x, N_y), dtype=np.int)
    grid[:, 0] = True
    grid[:, -1] = True

    return grid == 1


@jit
def f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s):
    f_eq = np.zeros((l, N_y, q))
    u_dot_c = np.zeros((l, N_y, q))

    u_dot_u = ux**2 + uy**2
    for i in range(q):
        u_dot_c[:, :, i] = ux * c_i[i, 0] + uy * c_i[i, 1]
        inner = 1 + (u_dot_c[:, :, i] / c_s**2) + (u_dot_c[:, :, i]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2))
        f_eq[:, :, i] = w_i[i] * rho * inner

    return f_eq


def temperature(T_dim, alpha, dx, ux, uy, T_BC_lower, T_BC_upper):
    T_dim_new = np.zeros((l, N_y))
    T_dim_new[:, 0] = T_BC_lower
    T_dim_new[:, -1] = T_BC_upper

    a = alpha / dx**2

    for j in range(1, N_y-1):
        for i in range(1, l - 1):
            T_dim_new[i, j] = (a - ux[i, j] / (2 * dx)) * T_dim[i-1, j] + (1 - 4 * a) * T_dim[i, j] \
                            + (a - ux[i, j] / (2 * dx)) * T_dim[i+1, j] + (a - uy[i, j] / (2 * dx)) * T_dim[i, j-1] \
                            + (a - uy[i, j] / (2 * dx)) * T_dim[i, j+1]

    # Periodic BCs
    T_dim_new[0, :] = T_dim_new[-2, :]
    T_dim_new[-1, :] = T_dim_new[1, :]

    return T_dim_new


# Temperature BCs
T_dim[:, 0] = T_BC_lower
T_dim[:, -1] = T_BC_upper

# Initialize boundary
bounds = pipe_boundary(l, N_y)

# Initialize equilibrium function
f_eq = f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s)
f_i = np.copy(f_eq)

for t in range(Nt):
    # Calculate macroscopic quantities
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # Calculate new T
    T_dim = temperature(T_dim, alpha_sim, dx_sim, ux, uy, T_BC_lower, T_BC_upper)

    # Buoyancy force
    F_buoy = T_dim[:, :, None] * g

    # View np.array
    idx = ["idx" for i in T_dim[1, :]]
    col = ["col" for j in T_dim[:, 1]]
    dataset = pd.DataFrame(T_dim.T, index=idx, columns=col)
    # print(dataset)

    # Calculate equilibrium distribution
    f_eq = f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s)

    ### Collision step
    f_star = f_i * (1 - dt_sim / tau) + f_eq * dt_sim / tau + dt_sim * F_pressure + dt_sim * F_buoy

    # Periodic boundary conditions
    f_star[0, :, 1] = f_star[-2, :, 1]
    f_star[0, :, 5] = f_star[-2, :, 5]
    f_star[0, :, 8] = f_star[-2, :, 8]

    f_star[-1, :, 3] = f_star[1, :, 3]
    f_star[-1, :, 6] = f_star[1, :, 6]
    f_star[-1, :, 7] = f_star[1, :, 7]

    ### Streaming step
    for j in range(1, N_y-1):
        for i in range(1, l - 1):
            f_i[i, j, 0] = f_star[i, j, 0]
            f_i[i, j, 1] = f_star[i-1, j, 1]
            f_i[i, j, 2] = f_star[i, j-1, 2]
            f_i[i, j, 3] = f_star[i+1, j, 3]
            f_i[i, j, 4] = f_star[i, j+1, 4]
            f_i[i, j, 5] = f_star[i-1, j-1, 5]
            f_i[i, j, 6] = f_star[i+1, j-1, 6]
            f_i[i, j, 7] = f_star[i+1, j+1, 7]
            f_i[i, j, 8] = f_star[i-1, j+1, 8]

            # Bounce-back BCs
            if j == N_y-2:  # No-slip BCs
                f_i[i, j, 4] = f_star[i, j, 2]
                f_i[i, j, 7] = f_star[i, j, 5]
                f_i[i, j, 8] = f_star[i, j, 6]
            elif j == 1:
                f_i[i, j, 2] = f_star[i, j, 4]
                f_i[i, j, 5] = f_star[i, j, 7]
                f_i[i, j, 6] = f_star[i, j, 8]

    if t in [999]:
        r = np.linspace(-N_y/2, N_y/2, num=N_y)
        r[0] = r[0] + (r[1] - r[0]) / 2
        r[-1] = r[-1] + (r[-2] - r[-1]) / 2
        r_phys = r*Cy
        R = r_phys[-1]
        umax_sim = np.amax(ux[np.int(np.rint(l / 2)), 1:N_y])
        u_th = umax_sim * (1 - r_phys ** 2 / R ** 2)

        ## Line plot
        plt.figure(np.int(t/200))
        plt.plot(ux[np.int(np.rint(l / 2)), :], r_phys)
        plt.plot(u_th, r_phys, 'o')
        plt.title('Velocity profile of simple Poiseuille flow.')
        plt.xlabel('$u$ (m/s)')
        plt.ylabel('$r$ (m)')
        plt.show()
        # plt.savefig("Figures/Pois_temp/lineplot_temp" + str(t) + ".png")

        # ## Vector plot
        # plt.figure(np.int(t/200))
        # plt.quiver(ux.T, uy.T)
        # plt.xlabel('$x$ (# lattice nodes)')
        # plt.ylabel('$y$ (# lattice nodes)')
        # plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
        # # plt.legend('Velocity vector')
        # plt.savefig("Figures/Pois_temp/arrowplot_temp" + str(t) + ".png")
        #
        # ## Heatmaps
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(ux.T, cmap=cm.Blues)
        # plt.colorbar()
        # plt.savefig("Figures/Pois_temp/heatmapx_temp" + str(t) + ".png")
        # #
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
        # plt.colorbar()
        # plt.savefig("Figures/Pois_temp/heatmap_uy_temp" + str(t) + ".png")
        #
        # ## Temperature heatmap
        # plt.figure(4)
        # plt.clf()
        # plt.imshow(np.flip(T_dim, axis=1).T, cmap=cm.Blues)
        # plt.colorbar()
        # plt.savefig("Figures/Pois_temp/heatmap_T" + str(t / 100) + ".png")

        # ## Line plot
        # plt.figure(5)
        # r = np.linspace(-N_y/2, N_y/2, num=N_y)
        # r_phys = r*Cy
        # T_phys = T_dim / beta_p + T0
        # plt.xlabel('$T$ (K)')
        # plt.ylabel('$r$ (m)')
        # plt.title('Temperature conduction in pipe with hot $(r=-R)$ and cold boundary $(r=R)$.\n No buoyancy.')
        # plt.plot(T_phys[np.int(np.rint(N_x / 2)), :], r_phys)
        # plt.savefig("Figures/Pois_temp/lineplot_T" + str(t / 100) + ".png")

# plt.plot(ux_ana, r_phys)
# plt.show()
