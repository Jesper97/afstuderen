import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import sys
from matplotlib import cm
from numba import jit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Define constants
# Physical parameters
L = 0.01            # m
H = 0.01            # m
g = 9.81            # m/s^2
T = 2               # s
nu = 1e-6           # m^2/s
alpha = 1.44e-7     # m^2/s
rho0 = 1e3          # kg/m^3
beta = 210e-6       # 1/K
T0 = 293            # K
T_C = 287           # K
T_H = 299           # K
umax = np.sqrt(g * beta * (T_H - T0) * L)

# Dimensionless numbers
Re = umax * H / nu
Ra = beta * (T_H - T0) * g * L**3 / (nu * alpha)
Pr = 7
Ma = 0.1

# Chose simulation parameters to determine the dependent one
rho0_sim = 1
tau = 0.7
Ny = 100

dx_sim = 1      # simulation length
dt_sim = 1      # simulation time
c_s = (1 / np.sqrt(3)) * (dx_sim / dt_sim)  # speed of sound
nu_sim = c_s**2 * (tau - 1/2)

# def check_stability(umax_sim, tau):
#     if umax_sim > 0.1:
#         print('Simulation velocity is', umax_sim, 'and might give unstable simulations.')
#     if tau < 1/2:
#         print('Relaxation time is', tau, 'and might give unstable simulations.')

# Determine dependent parameter
umax_sim = Re * nu_sim / Ny
print('umax_sim', umax_sim)
# check_stability(umax_sim, tau)

# Calculate conversion parameters
dx = H / Ny
dt = c_s**2 * (tau - 1/2) * dx**2 / nu
print('dt', dt)
Cu = dx / dt
Cg = dx / dt**2
Crho = rho0 / rho0_sim
CF = Crho * Cg
print('CF', CF)

# D2Q9 lattice constants
c_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int)
c_opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int)
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
q = 9                               # number of directions

# Simulation parameters
alpha_sim = nu_sim / Pr

# Grid and time steps
Nx = Ny                     # lattice nodes in the x-direction
Nt = np.int(T / dt)         # time steps

# Forces
g_sim = g / Cg
gi_sim = (g_sim * w_i / c_s**2) * np.array([0, 0, -1, 0, 1, -1, -1, 1, 1])

# Initial conditions
ux = np.zeros((Nx, Ny))       # Simulation velocity in x direction
uy = np.zeros((Nx, Ny))       # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones((Nx, Ny))   # Simulation density
T_dim = np.zeros((Nx, Ny))    # Dimensionless simulation temperature

# Temperature BCS
T_BC_upper = np.ones(Ny) * beta * (T_H - T0)
T_BC_lower = np.ones(Ny) * beta * (T_C - T0)


@jit
def fluid(Nx, Ny, f_i, f_star):
    for i in range(1, Nx-1):
        f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
        f_i[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
        f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
        f_i[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
        f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
        f_i[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 5]
        f_i[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
        f_i[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
        f_i[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

    return f_i

@jit
def left_wall(Ny, f_i, f_star):
    i = 0

    f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f_i[i, 1:Ny-1, 1] = f_star[i, 1:Ny-1, 3]        # Bounce
    f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f_i[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
    f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f_i[i, 1:Ny-1, 5] = f_star[i, 1:Ny-1, 7]        # Bounce
    f_i[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
    f_i[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
    f_i[i, 1:Ny-1, 8] = f_star[i, 1:Ny-1, 6]        # Bounce

    return f_i

@jit
def right_wall(Nx, Ny, f_i, f_star):
    i = Nx - 1

    f_i[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f_i[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
    f_i[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f_i[i, 1:Ny-1, 3] = f_star[i, 1:Ny-1, 1]        # Bounce
    f_i[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f_i[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 5]
    f_i[i, 1:Ny-1, 6] = f_star[i, 1:Ny-1, 8]        # Bounce
    f_i[i, 1:Ny-1, 7] = f_star[i, 1:Ny-1, 5]        # Bounce
    f_i[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

    return f_i

@jit
def lower_wall(Nx, f_i, f_star):
    j = 0

    f_i[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f_i[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f_i[1:Nx-1, j, 2] = f_star[1:Nx-1, j, 4]        # Bounce
    f_i[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f_i[1:Nx-1, j, 4] = f_star[1:Nx-1, j+1, 4]
    f_i[1:Nx-1, j, 5] = f_star[1:Nx-1, j, 7]        # Bounce
    f_i[1:Nx-1, j, 6] = f_star[1:Nx-1, j, 8]        # Bounce
    f_i[1:Nx-1, j, 7] = f_star[2:Nx, j+1, 7]
    f_i[1:Nx-1, j, 8] = f_star[0:Nx-2, j+1, 8]

    return f_i

@jit
def upper_wall(Nx, Ny, f_i, f_star):
    j = Ny - 1

    f_i[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f_i[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f_i[1:Nx-1, j, 2] = f_star[1:Nx-1, j-1, 2]
    f_i[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f_i[1:Nx-1, j, 4] = f_star[1:Nx-1, j, 2]        # Bounce
    f_i[1:Nx-1, j, 5] = f_star[0:Nx-2, j-1, 5]
    f_i[1:Nx-1, j, 6] = f_star[2:Nx, j-1, 6]
    f_i[1:Nx-1, j, 7] = f_star[1:Nx-1, j, 5]        # Bounce
    f_i[1:Nx-1, j, 8] = f_star[1:Nx-1, j, 6]        # Bounce

    return f_i

@jit
def lower_left_corner(f_i, f_star):
    i = 0
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j+1, 4]
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i+1, j+1, 7]
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@jit
def lower_right_corner(Nx, f_i, f_star):
    i = Nx - 1
    j = 0

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j, 4]                  # Bounce
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j+1, 4]
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i-1, j+1, 8]

    return f_i

@jit
def upper_left_corner(f_i, f_star):
    i = 0
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i, j, 3]                  # Bounce
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i+1, j, 3]
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    f_i[i, j, 5] = f_star[i, j, 7]                  # Bounce
    f_i[i, j, 6] = f_star[i+1, j-1, 6]
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@jit
def upper_right_corner(f_i, f_star):
    i = Nx - 1
    j = Ny - 1

    f_i[i, j, 0] = f_star[i, j, 0]
    f_i[i, j, 1] = f_star[i-1, j, 1]
    f_i[i, j, 2] = f_star[i, j-1, 2]
    f_i[i, j, 3] = f_star[i, j, 1]                  # Bounce
    f_i[i, j, 4] = f_star[i, j, 2]                  # Bounce
    f_i[i, j, 5] = f_star[i-1, j-1, 5]
    f_i[i, j, 6] = f_star[i, j, 8]                  # Bounce
    f_i[i, j, 7] = f_star[i, j, 5]                  # Bounce
    f_i[i, j, 8] = f_star[i, j, 6]                  # Bounce

    return f_i

@jit
def f_equilibrium(w_i, rho, ux, uy, c_i, q, c_s):
    f_eq = np.zeros((Nx, Ny, q))
    u_dot_c = np.zeros((Nx, Ny, q))

    u_dot_u = ux**2 + uy**2
    for i in range(q):
        u_dot_c[:, :, i] = ux * c_i[i, 0] + uy * c_i[i, 1]
        inner = 1 + (u_dot_c[:, :, i] / c_s**2) + (u_dot_c[:, :, i]**2 / (2 * c_s**4)) - (u_dot_u / (2 * c_s**2))
        f_eq[:, :, i] = w_i[i] * rho * inner

    return f_eq

def streaming(Nx, Ny, f_i, f_star):
    f_i = fluid(Nx, Ny, f_i, f_star)
    f_i = left_wall(Ny, f_i, f_star)
    f_i = right_wall(Nx, Ny, f_i, f_star)
    f_i = lower_wall(Nx, f_i, f_star)
    f_i = upper_wall(Nx, Ny, f_i, f_star)
    f_i = lower_left_corner(f_i, f_star)
    f_i = lower_right_corner(Nx, f_i, f_star)
    f_i = upper_left_corner(f_i, f_star)
    f_i = upper_right_corner(f_i, f_star)

    return f_i

def temperature(T_dim, alpha, dx, ux, uy, T_BC_lower, T_BC_upper):
    T_dim_new = np.zeros((Nx, Ny))
    T_dim_new[0, :] = T_BC_upper
    T_dim_new[-1, :] = T_BC_lower

    a = alpha / dx**2

    for j in range(1, Ny-1):
        for i in range(1, Nx - 1):
            T_dim_new[i, j] = (a - ux[i, j] / (2 * dx)) * T_dim[i-1, j] + (1 - 4 * a) * T_dim[i, j] \
                            + (a - ux[i, j] / (2 * dx)) * T_dim[i+1, j] + (a - uy[i, j] / (2 * dx)) * T_dim[i, j-1] \
                            + (a - uy[i, j] / (2 * dx)) * T_dim[i, j+1]

    # Periodic BCs
    T_dim_new[0, :] = T_dim_new[-2, :]
    T_dim_new[-1, :] = T_dim_new[1, :]

    return T_dim_new


# Temperature BCs
T_dim[0, :] = T_BC_lower
T_dim[-1, :] = T_BC_upper

# Initialize equilibrium function
f_eq = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)
f_i = np.copy(f_eq)

for t in range(Nt):
    # Calculate macroscopic quantities
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # Calculate new T
    T_dim = temperature(T_dim, alpha_sim, dx_sim, ux, uy, T_BC_lower, T_BC_upper)

    # Buoyancy force
    Fi_buoy = T_dim[:, :, None] * gi_sim

    # # View np.array
    # idx = ["idx" for i in T_dim[1, :]]
    # col = ["col" for j in T_dim[:, 1]]
    # dataset = pd.DataFrame(T_dim.T, index=idx, columns=col)
    # # print(dataset)

    # Calculate equilibrium distribution
    f_eq = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)

    ### Collision step
    f_star = f_i * (1 - dt_sim / tau) + f_eq * dt_sim / tau + dt_sim * Fi_buoy

    ### Streaming step
    f_i = streaming(Nx, Ny, f_i, f_star)

r = np.linspace(-Ny/2, Ny/2, num=Ny)
r[0] = r[0] + (r[1] - r[0]) / 2
r[-1] = r[-1] + (r[-2] - r[-1]) / 2
r_phys = r*dx
R = r_phys[-1]
umax_sim = np.amax(ux[np.int(np.rint(Nx / 2)), 1:Ny])
u_th = umax_sim * (1 - r_phys ** 2 / R ** 2)

## Vector plot
# plt.figure(np.int(t/200)+1)
x = np.linspace(0, L, len(ux))
y = np.linspace(0, H, len(uy))
# fig = ff.create_streamline(x, y, ux.T, uy.T)
# fig.show()
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
# plt.legend('Velocity vector')
# plt.savefig("Figures/sq_cav_th/arrowplot_temp" + str(t-2) + ".png")

## Vector plot
plt.figure(np.int(t/200)+1, dpi=300)
plt.quiver(ux.T, uy.T)
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
# plt.legend('Velocity vector')
plt.savefig("Figures/sq_cav_th/arrowplot_temp" + str(t-1) + ".png")

## Heatmaps
plt.figure(2)
plt.clf()
plt.imshow(ux.T, cmap=cm.Blues, origin='lower')
plt.colorbar()
plt.savefig("Figures/sq_cav_th/heatmapx_temp" + str(t) + ".png")
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
