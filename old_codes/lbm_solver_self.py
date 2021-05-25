import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import sys
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

steps = 100000
umax = 0.1
nx = 128  # by convention, dx = dy = dt = 1.0 (lattice units)
ny = nx
niu = 0.0128
tau = 3.0 * niu + 0.5
inv_tau = 1.0 / tau
rho = np.ones((nx, ny), dtype=np.float32)
vel = np.zeros((nx, ny, 2), dtype=np.float32)
f_old = np.zeros((nx, ny, 9), dtype=np.float32)
f_new = np.zeros((nx, ny, 9), dtype=np.float32)
bc_value = np.array([[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
w = np.array([ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


@njit
def f_eq(i, j, k, rho, vel):
    uc = e[k, 0] * vel[i, j, 0] + e[k, 1] * vel[i, j, 1]
    uv = vel[i, j, 0]**2 + vel[i, j, 1]**2

    return w[k] * rho[i, j] * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * uv)


@njit
def initialize(vel, rho, f_new, f_old):
    for j in range(rho.shape[1]):
        for i in range(rho.shape[0]):
            vel[i, j, 0] = 0.0
            vel[i, j, 1] = 0.0
            rho[i, j] = 1.0

            for k in range(9):
                f_new[i, j, k] = f_eq(i, j, k, rho, vel)
                f_old[i, j, k] = f_new[i, j, k]

    return vel, rho, f_new, f_old


@njit
def streaming_and_collision(r, u, f_old, f_new):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            for k in range(9):
                ip = i - e[k, 0]
                jp = j - e[k, 1]
                f_new[i, j, k] = (1.0 - inv_tau) * f_old[ip, jp, k] + f_eq(ip, jp, k, r, u) * inv_tau

    # Left wall
    for j in range(1, ny-1):
        i = 0
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i, j, 3] + f_eq(i, j, 3, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j-1, 2] + f_eq(i, j-1, 2, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i+1, j, 3] + f_eq(i+1, j, 3, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j+1, 4] + f_eq(i, j+1, 4, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i, j, 7] + f_eq(i, j, 7, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i+1, j-1, 6] + f_eq(i+1, j-1, 6, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i+1, j+1, 7] + f_eq(i+1, j+1, 7, r, u)*inv_tau
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i, j, 6] + f_eq(i, j, 6, r, u)*inv_tau

        # Right wall
        i = nx - 1
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i-1, j, 1] + f_eq(i-1, j, 1, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j-1, 2] + f_eq(i, j-1, 2, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i, j, 1] + f_eq(i, j, 1, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j+1, 4] + f_eq(i, j+1, 4, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i-1, j-1, 5] + f_eq(i-1, j-1, 5, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i, j, 8] + f_eq(i, j, 8, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i, j, 5] + f_eq(i, j, 5, r, u)*inv_tau
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i-1, j+1, 8] + f_eq(i-1, j+1, 8, r, u)*inv_tau

    # Bottom wall
    for i in range(1, nx-1):
        j = 0
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i-1, j, 1] + f_eq(i-1, j, 1, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j, 4] + f_eq(i, j, 4, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i+1, j, 3] + f_eq(i+1, j, 3, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j+1, 4] + f_eq(i, j+1, 4, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i, j, 7] + f_eq(i, j, 7, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i, j, 8] + f_eq(i, j, 8, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i+1, j+1, 7] + f_eq(i+1, j+1, 7, r, u)*inv_tau
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i-1, j+1, 8] + f_eq(i-1, j+1, 8, r, u)*inv_tau

        # Top wall
        j = nx - 1
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i-1, j, 1] + f_eq(i-1, j, 1, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j-1, 2] + f_eq(i, j-1, 2, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i+1, j, 3] + f_eq(i+1, j, 3, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j, 2] + f_eq(i, j, 2, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i-1, j-1, 5] + f_eq(i-1, j-1, 5, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i+1, j-1, 6] + f_eq(i+1, j-1, 6, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i, j, 5] + f_eq(i, j, 5, r, u)*inv_tau - \
                              6 * w[5] * r[i, j] * bc_value[1, 0]
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i, j, 6] + f_eq(i, j, 6, r, u)*inv_tau + \
                              6 * w[6] * r[i, j] * bc_value[1, 0]

        # Bottom left corner
        i = 0
        j = 0
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i, j, 3] + f_eq(i, j, 3, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j, 4] + f_eq(i, j, 4, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i+1, j, 3] + f_eq(i+1, j, 3, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j+1, 4] + f_eq(i, j+1, 4, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i, j, 7] + f_eq(i, j, 7, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i, j, 8] + f_eq(i, j, 8, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i+1, j+1, 7] + f_eq(i+1, j+1, 7, r, u)*inv_tau
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i, j, 6] + f_eq(i, j, 6, r, u)*inv_tau

        # Bottom right corner
        i = nx - 1
        j = 0
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i-1, j, 1] + f_eq(i-1, j, 1, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j, 4] + f_eq(i, j, 4, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i, j, 1] + f_eq(i, j, 1, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j+1, 4] + f_eq(i, j+1, 4, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i, j, 7] + f_eq(i, j, 7, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i, j, 8] + f_eq(i, j, 8, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i, j, 5] + f_eq(i, j, 5, r, u)*inv_tau
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i-1, j+1, 8] + f_eq(i-1, j+1, 8, r, u)*inv_tau

        # Top right corner
        i = nx - 1
        j = ny - 1
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i-1, j, 1] + f_eq(i-1, j, 1, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j-1, 2] + f_eq(i, j-1, 2, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i, j, 1] + f_eq(i, j, 1, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j, 2] + f_eq(i, j, 2, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i-1, j-1, 5] + f_eq(i-1, j-1, 5, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i, j, 8] + f_eq(i, j, 8, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i, j, 5] + f_eq(i, j, 5, r, u)*inv_tau - \
                                  6 * w[5] * r[i, j] * bc_value[1, 0]
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i, j, 6] + f_eq(i, j, 6, r, u)*inv_tau + \
                                  6 * w[6] * r[i, j] * bc_value[1, 0]

        # Top left corner
        i = 0
        j = ny - 1
        f_new[i, j, 0] = (1.0-inv_tau)*f_old[i, j, 0] + f_eq(i, j, 0, r, u)*inv_tau
        f_new[i, j, 1] = (1.0-inv_tau)*f_old[i, j, 3] + f_eq(i, j, 3, r, u)*inv_tau
        f_new[i, j, 2] = (1.0-inv_tau)*f_old[i, j-1, 2] + f_eq(i, j-1, 2, r, u)*inv_tau
        f_new[i, j, 3] = (1.0-inv_tau)*f_old[i+1, j, 3] + f_eq(i+1, j, 3, r, u)*inv_tau
        f_new[i, j, 4] = (1.0-inv_tau)*f_old[i, j, 2] + f_eq(i, j, 2, r, u)*inv_tau
        f_new[i, j, 5] = (1.0-inv_tau)*f_old[i, j, 7] + f_eq(i, j, 7, r, u)*inv_tau
        f_new[i, j, 6] = (1.0-inv_tau)*f_old[i+1, j-1, 6] + f_eq(i+1, j-1, 6, r, u)*inv_tau
        f_new[i, j, 7] = (1.0-inv_tau)*f_old[i, j, 5] + f_eq(i, j, 5, r, u)*inv_tau - \
                                  6 * w[5] * r[i, j] * bc_value[1, 0]
        f_new[i, j, 8] = (1.0-inv_tau)*f_old[i, j, 6] + f_eq(i, j, 6, r, u)*inv_tau + \
                                  6 * w[6] * r[i, j] * bc_value[1, 0]

    return f_new


@njit
def update_macro_var(rho, vel, f_old, f_new):
    for j in range(0, ny):
        for i in range(0, nx):
            rho[i, j] = 0.0
            vel[i, j, 0] = 0.0
            vel[i, j, 1] = 0.0

            for k in range(9):
                f_old[i, j, k] = f_new[i, j, k]
                rho[i, j] += f_new[i, j, k]
                vel[i, j, 0] += e[k, 0] * f_new[i, j, k]
                vel[i, j, 1] += e[k, 1] * f_new[i, j, k]

            vel[i, j, 0] /= rho[i, j]
            vel[i, j, 1] /= rho[i, j]

    return rho, vel, f_old


@njit
def solve(vel, rho, f_new, f_old):
    vel, rho, f, f_old = initialize(vel, rho, f_new, f_old)

    for i in range(steps):
        f = streaming_and_collision(rho, vel, f_old, f)
        rho, vel, f_old = update_macro_var(rho, vel, f_old, f)

        if i % 1000 == 0:
            print(i)
        #     easy_view("ux", vel[:, :, 0])

    # easy_view("ux", vel[:, :, 0])
    # easy_view(1, f[:, :, 1])
    # easy_view(2, f[:, :, 2])
    # easy_view(3, f[:, :, 3])
    # easy_view(4, f[:, :, 4])
    # easy_view(5, f[:, :, 5])
    # easy_view(6, f[:, :, 6])
    # easy_view(7, f[:, :, 7])
    # easy_view(8, f[:, :, 8])

    return vel


u = solve(vel, rho, f_new, f_old)

# Compare results with literature
y_ref, u_ref = np.loadtxt('/Users/Jesper/Documents/MEP/Code/Working code/data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 2))
x_ref, v_ref = np.loadtxt('/Users/Jesper/Documents/MEP/Code/Working code/data/ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 8))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(u[nx // 2, :, 0] / umax, np.linspace(0, 1.0, nx), 'b-', label='LBM')
axes.plot(u_ref, y_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/ux_Re1000_test1.png")

plt.clf()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(np.linspace(0, 1.0, nx), u[:, nx // 2, 1] / umax, 'b-', label='LBM')
axes.plot(x_ref, v_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/uy_Re1000_test1.png")
