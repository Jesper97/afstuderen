import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit
from matplotlib import cm

np.set_printoptions(threshold=sys.maxsize)

# Define constants
# Grid and time steps
N_x = 81        # lattice nodes in the x-direction
N_y = 21        # lattice nodes in the x-direction
N_t = 1000       # time steps

# Simulation parameters
dx = 1          # simulation length
dt = 1          # simulation time

# Simulation constants
c_s = (1 / np.sqrt(3)) * (dx / dt)  # speed of sound
q = 9                               # number of directions

# Material simulation constants
tau = 0.9

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Physical constants
L = 0.2     # length of pipe
H = 0.2     # height of pipe
T = 20
nu = c_s**2 * (tau - dt / 2)
F = 0.001 * w_i * np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])

# Conversion factors
Cx = H / N_x
Cy = L / N_y
Ct = T / N_t

# Initial conditions
u_x = np.zeros((N_x, N_y))
u_y = np.zeros((N_x, N_y))
rho = np.ones((N_x, N_y))

# Create boundary mask
def pipe_boundary(N_x, N_y):

    grid = np.zeros((N_x, N_y), dtype=np.int)
    grid[:, 0] = True
    grid[:, -1] = True

    return grid == 1


@jit
def f_equilibrium(w_i, rho, u_x, u_y, c_i, q, c_s, F):
    f_eq = np.zeros((N_x, N_y, q))
    u_dot_c = np.zeros((N_x, N_y, q))

    u_dot_u = u_x**2 + u_y**2
    for i in range(q):
        u_dot_c[:, :, i] = u_x * c_i[i, 0] + u_y * c_i[i, 1]
        inner = 1 + (u_dot_c[:, :, i] / c_s**2) + (u_dot_c[:, :, i]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2))
        f_eq[:, :, i] = w_i[i] * rho * inner + F[i]

    return f_eq


# Initialize boundary
bounds = pipe_boundary(N_x, N_y)

# Initialize equilibrium function
f_eq = f_equilibrium(w_i, rho, u_x, u_y, c_i, q, c_s, F)
f_i = np.copy(f_eq)

for t in range(N_t):
    # Calculate macroscopic quantities
    rho = np.sum(f_i, axis=2)
    u_x = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    u_y = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # Calculate equilibrium distribution
    f_eq = f_equilibrium(w_i, rho, u_x, u_y, c_i, q, c_s, F)

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

    if t in [998, 1999]:
        r = np.linspace(-N_y/2, N_y/2, num=N_y)
        r[0] = r[0] + (r[1] - r[0]) / 2
        r[-1] = r[-1] + (r[-2] - r[-1]) / 2
        r_phys = r*Cy
        R = r_phys[-1]
        u_max = np.amax(u_x[np.int(np.rint(N_x/2)), 1:N_y])
        u_th = u_max * (1 - r_phys**2 / R**2)

        ## Line plot
        plt.figure(0)
        plt.plot(u_x[np.int(np.rint(N_x/2)), :], r_phys)
        # plt.plot(u_x[1, :], r_phys)
        plt.plot(u_th, r_phys, 'o')
        plt.savefig("Figures/Pois_test/lineplot" + str(t / 100) + ".png")

        ## Vector plot
        plt.figure(1)
        plt.quiver(u_x.T, u_y.T)
        plt.savefig("Figures/Pois_test/arrowplot" + str(t / 100) + ".png")

        ## Heatmaps
        plt.figure(2)
        plt.clf()
        plt.imshow(u_x.T, cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("Figures/Pois_test/heatmapx" + str(t / 100) + ".png")

        plt.figure(3)
        plt.clf()
        plt.imshow(np.flip(u_y, axis=1).T, cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("Figures/Pois_test/heatmapy" + str(t / 100) + ".png")

# plt.plot(ux_ana, r_phys)
# plt.show()
