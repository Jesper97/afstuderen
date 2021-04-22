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
T = 2               # s
nu = 1e-6          # m^2/s
rho0 = 1e3           # kg/m^3
dp = 0.01            # kg/(m s^2)
Fp = dp / L         # Pressure force density (kg/(m^2 s^2))
umax = 0.005

# Dimensionless numbers
Re = umax * H / nu
print('Re', Re)
Pr = 7
Ra = 10e6
Ma = 0.1

# Chose simulation parameters to determine the dependent one
rho0_sim = 1
tau = 0.9
Ny = 80

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
umax_sim = Re * nu_sim / Ny
print('umax_sim', umax_sim)
check_stability(umax_sim, tau)

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
print('alpha_sim', alpha_sim)

# Grid and time steps
Nx = Ny                               # lattice nodes in the x-direction
Nt = np.int(T / dt)       # time steps

# Forces
Fp_sim = Fp / CF
Fpi_sim = (Fp_sim * w_i / c_s**2) * np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
print(Fpi_sim)
g_sim = g / Cg
gi_sim = (g_sim * w_i / c_s**2) * np.array([0, 0, -1, 0, 1, -1, -1, 1, 1])
print(gi_sim)

# Initial conditions
ux = np.zeros((Nx, Ny))       # Simulation velocity in x direction
uy = np.zeros((Nx, Ny))       # Simulation velocity in y direction
rho_sim = rho0_sim * np.ones((Nx, Ny))   # Simulation density
T_dim = np.zeros((Nx, Ny))    # Dimensionless simulation temperature

# Temperature BCS
beta_p = 210e-6
T0 = 293
T_up_p = 289
T_down_p = 297
T_BC_upper = np.ones(Nx) * beta_p * (T_up_p - T0)
T_BC_lower = np.ones(Nx) * beta_p * (T_down_p - T0)

for node in range(len(T_BC_upper)):
    if node <= len(T_BC_upper)/2:
        T_BC_lower[node] *= -1


# Create boundary mask
def pipe_boundary(Nx, Ny):

    grid = np.zeros((Nx, Ny), dtype=np.int)
    grid[:, 0] = True
    grid[:, -1] = True

    return grid == 1


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


def temperature(T_dim, alpha, dx, ux, uy, T_BC_lower, T_BC_upper):
    T_dim_new = np.zeros((Nx, Ny))
    T_dim_new[:, 0] = T_BC_lower
    T_dim_new[:, -1] = T_BC_upper

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
T_dim[:, 0] = T_BC_lower
T_dim[:, -1] = T_BC_upper

# Initialize boundary
bounds = pipe_boundary(Nx, Ny)

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

    # View np.array
    idx = ["idx" for i in T_dim[1, :]]
    col = ["col" for j in T_dim[:, 1]]
    dataset = pd.DataFrame(T_dim.T, index=idx, columns=col)
    # print(dataset)

    # Calculate equilibrium distribution
    f_eq = f_equilibrium(w_i, rho_sim, ux, uy, c_i, q, c_s)

    ### Collision step
    f_star = f_i * (1 - dt_sim / tau) + f_eq * dt_sim / tau + dt_sim * Fpi_sim + dt_sim * Fi_buoy

    # Periodic boundary conditions
    f_star[0, :, 1] = f_star[-2, :, 1]
    f_star[0, :, 5] = f_star[-2, :, 5]
    f_star[0, :, 8] = f_star[-2, :, 8]

    f_star[-1, :, 3] = f_star[1, :, 3]
    f_star[-1, :, 6] = f_star[1, :, 6]
    f_star[-1, :, 7] = f_star[1, :, 7]

    ### Streaming step
    for j in range(1, Ny-1):
        for i in range(1, Nx - 1):
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
            if j == Ny-2:  # No-slip BCs
                f_i[i, j, 4] = f_star[i, j, 2]
                f_i[i, j, 7] = f_star[i, j, 5]
                f_i[i, j, 8] = f_star[i, j, 6]
            elif j == 1:
                f_i[i, j, 2] = f_star[i, j, 4]
                f_i[i, j, 5] = f_star[i, j, 7]
                f_i[i, j, 6] = f_star[i, j, 8]

r = np.linspace(-Ny/2, Ny/2, num=Ny)
r[0] = r[0] + (r[1] - r[0]) / 2
r[-1] = r[-1] + (r[-2] - r[-1]) / 2
r_phys = r*dx
R = r_phys[-1]
umax_sim = np.amax(ux[np.int(np.rint(Nx / 2)), 1:Ny])
u_th = umax_sim * (1 - r_phys ** 2 / R ** 2)

## Vector plot
# plt.figure(np.int(t/200)+1)
x = np.linspace(dx, L-dx, len(ux))
y = np.linspace(dx, H-dx, len(uy))
# fig = ff.create_streamline(x, y, ux.T, uy.T)
# fig.show()
# plt.xlabel('$x$ (# lattice nodes)')
# plt.ylabel('$y$ (# lattice nodes)')
# plt.title('Velocity profile in pipe with hot plate for $x < L/2$ and cold plate for $x > L/2$. \n $p>0$')
# plt.legend('Velocity vector')
# plt.savefig("Figures/sq_cav_th/arrowplot_temp" + str(t-2) + ".png")

# Vector plot
plt.figure(np.int(t/200)+2, dpi=300)
plt.quiver(Cu*ux.T, Cu*uy.T)
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'$u$ in gravity Poiseuille flow')
# plt.legend('Velocity vector')
plt.savefig(f"Figures/vert_pois/arrowplot_time{Time}.png")

# Heatmaps
plt.figure(2)
plt.clf()
plt.imshow(Cu*ux.T, cmap=cm.Blues, origin='lower')
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'$u_x$ in gravity Poiseuille flow')
plt.colorbar()
plt.savefig(f"Figures/vert_pois/heatmap_ux_time{Time}.png")

plt.figure(3)
plt.clf()
plt.imshow(np.flip(Cu*uy, axis=1).T, cmap=cm.Blues)
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'$u_y$ in gravity Poiseuille flow')
plt.colorbar()
plt.savefig(f"Figures/vert_pois/heatmap_uy_time{Time}.png")

plt.figure(5)
plt.clf()
plt.imshow(np.flip(rho, axis=1).T, cmap=cm.Blues)
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'$\\rho$ in gravity Poiseuille flow')
plt.colorbar()
plt.savefig(f"Figures/vert_pois/heatmap_rho_time{Time}.png")

# ## Line plot
# plt.figure(5)
# r = np.linspace(-Ny/2, Ny/2, num=Ny)
# r_phys = r*Cy
# T_phys = T_dim / beta_p + T0
# plt.xlabel('$T$ (K)')
# plt.ylabel('$r$ (m)')
# plt.title('Temperature conduction in pipe with hot $(r=-R)$ and cold boundary $(r=R)$.\n No buoyancy.')
# plt.plot(T_phys[np.int(np.rint(N_x / 2)), :], r_phys)
# plt.savefig("Figures/Pois_temp/lineplot_T" + str(t / 100) + ".png")

# plt.plot(ux_ana, r_phys)
# plt.show()
