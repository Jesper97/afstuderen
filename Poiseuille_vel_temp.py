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

# Grid and time steps
N_x = 40        # lattice nodes in the x-direction
N_y = 20        # lattice nodes in the x-direction
N_t = 2000       # time steps

# Simulation parameters
dx = 1          # simulation length
dt = 1          # simulation time
c_s = (1 / np.sqrt(3)) * (dx / dt)  # speed of sound
q = 9                               # number of directions
tau = 0.9
alpha = 0.1

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Physical constants
L = 0.2     # length of pipe
H = 0.2     # height of pipe
nu = c_s**2 * (tau - dt / 2)
F_pressure = 0.001 * w_i * np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
g = 0.001 * w_i * np.array([0, 0, -1, 0, 1, -1, -1, 1, 1])

# Conversion factors
Cy = 0.2 / N_y

# Initial conditions
ux = np.zeros((N_x, N_y))       # Simulation velocity in x direction
uy = np.zeros((N_x, N_y))       # Simulation velocity in y direction
rho = np.ones((N_x, N_y))       # Simulation density
T_dim = np.zeros((N_x, N_y))    # Dimensionless simulation temperature

# Temperature BCS
T_BC_upper = np.ones(N_x) * -0.001
T_BC_lower = np.ones(N_x) * 0.001


# Create boundary mask
def pipe_boundary(N_x, N_y):

    grid = np.zeros((N_x, N_y), dtype=np.int)
    grid[:, 0] = True
    grid[:, -1] = True

    return grid == 1


@jit
def f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s, F_pressure):
    f_eq = np.zeros((N_x, N_y, q))
    u_dot_c = np.zeros((N_x, N_y, q))

    u_dot_u = ux**2 + uy**2
    for i in range(q):
        u_dot_c[:, :, i] = ux * c_i[i, 0] + uy * c_i[i, 1]
        inner = 1 + (u_dot_c[:, :, i] / c_s**2) + (u_dot_c[:, :, i]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2))
        f_eq[:, :, i] = w_i[i] * rho * inner + F_pressure[i]

    return f_eq


def temperature(T_dim, alpha, dx, ux, uy, T_BC_lower, T_BC_upper):
    T_dim_new = np.zeros((N_x, N_y))
    T_dim_new[:, 0] = T_BC_lower
    T_dim_new[:, -1] = T_BC_upper

    a = alpha / dx**2

    for j in range(1, N_y-1):
        for i in range(1, N_x-1):
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
bounds = pipe_boundary(N_x, N_y)

# Initialize equilibrium function
f_eq = f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s, F_pressure)
f_i = np.copy(f_eq)

for t in range(N_t):
    # Calculate macroscopic quantities
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # Calculate new T
    T_dim = temperature(T_dim, alpha, dx, ux, uy, T_BC_lower, T_BC_upper)

    # View np.array
    idx = ["idx" for i in T_dim[1, :]]
    col = ["col" for j in T_dim[:, 1]]
    dataset = pd.DataFrame(T_dim.T, index=idx, columns=col)
    # print(dataset)

    # Calculate equilibrium distribution
    f_eq = f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s, F_pressure)

    ### Collision step
    f_star = f_i * (1 - dt / tau) + f_eq * dt / tau

    # Periodic boundary conditions
    f_star[0, :, 1] = f_star[-2, :, 1]
    f_star[0, :, 5] = f_star[-2, :, 5]
    f_star[0, :, 8] = f_star[-2, :, 8]

    f_star[-1, :, 3] = f_star[1, :, 3]
    f_star[-1, :, 6] = f_star[1, :, 6]
    f_star[-1, :, 7] = f_star[1, :, 7]

    ### Streaming step
    for j in range(1, N_y-1):
        for i in range(1, N_x-1):
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

    if t in [1999]:
        r = np.linspace(-N_y/2, N_y/2, num=N_y)
        r[0] = r[0] + (r[1] - r[0]) / 2
        r[-1] = r[-1] + (r[-2] - r[-1]) / 2
        r_phys = r*Cy
        R = r_phys[-1]
        u_max = np.amax(ux[np.int(np.rint(N_x / 2)), 1:N_y])
        u_th = u_max * (1 - r_phys**2 / R**2)

        # ## Line plot
        # plt.figure(0)
        # plt.plot(ux[np.int(np.rint(N_x / 2)), :], r_phys)
        # plt.plot(u_th, r_phys, 'o')
        # plt.savefig("Figures/Pois_temp/lineplot_temp" + str(t / 100) + ".png")
        #
        # ## Vector plot
        # plt.figure(1)
        # plt.quiver(ux.T, uy.T)
        # plt.savefig("Figures/Pois_temp/arrowplot_temp" + str(t / 100) + ".png")

        # ## Heatmaps
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(ux.T, cmap=cm.Blues)
        # plt.colorbar()
        # plt.savefig("Figures/Pois_temp/heatmapx_temp" + str(t / 100) + ".png")
        #
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(np.flip(uy, axis=1).T, cmap=cm.Blues)
        # plt.colorbar()
        # plt.savefig("Figures/Pois_temp/heatmapy_temp" + str(t / 100) + ".png")

        ## Temperature heatmap
        plt.figure(4)
        plt.clf()
        plt.imshow(np.flip(T_dim, axis=1).T, cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("Figures/Pois_temp/heatmap_T" + str(t / 100) + ".png")

        ## Line plot
        plt.figure(5)
        plt.plot(T_dim[np.int(np.rint(N_x / 2)), :], r_phys)
        plt.savefig("Figures/Pois_temp/lineplot_T" + str(t / 100) + ".png")

# plt.plot(ux_ana, r_phys)
# plt.show()
