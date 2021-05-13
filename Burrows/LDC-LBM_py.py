import matplotlib as plt
import numpy as np

# Parameters
TRT = True
aspect_ratio = 1
Nmsg = 5000
Nsteps = 1.0e8
q = 9
magic = 0.25

# Constants
Re = 1000
Nx = int(80)
Ny = int(aspect_ratio * Nx)
umax_sim = 0.1
nu_sim = umax_sim * Nx / Re
tau = 3 * nu_sim + 0.5
cs = 1.0 / np.sqrt(3)
Ma = umax_sim / cs
omega = 1.0 / tau
df0 = 1.0
omega_minus = 1.0 / (0.5 + magic / (tau - 0.5))

# Print Simulation Information
print("Simulating 2D Lid-driven cavity")
print("      domain size: %u x %u" % (Nx, Ny))
print("               nu: %g"      % nu_sim)
print("              tau: %g"      % tau)
print("            u_max: %g"      % umax_sim)
print("             Mach: %g"      % Ma)
print("    message every: %u steps" % Nmsg)

f0 = np.zeros((Nx, Ny), dtype=float)
f1 = np.zeros((Nx, Ny, q), dtype=float)
f2 = np.zeros((Nx, Ny, q), dtype=float)
rho = np.ones((Nx, Ny), dtype=float)
ux = np.ones((Nx, Ny), dtype=float)
uy = np.ones((Nx, Ny), dtype=float)

w0 = 4.0/9.0
ws = 1.0/9.0
wd = 1.0/9.0

# Field indices?

# Equilibrium distribution
def single_feq(w, rho, cdotu, wu2x15):
    return w * (rho + (3.0 * cdotu) + (4.5 * (cdotu * cdotu))) - wu2x15

def eq_dist_func(w0, ws, wd, f0, f1, r, u, v):
    drho = r - 1.0
    ux = u
    uy = v

    u2x15 = 1.5 * (ux * ux + uy * uy)
    wsu2x15 = ws * u2x15
    wdu2x15 = wd * u2x15

    f0[:, :] = w0 * (drho - u2x15)
    f1[:, :, 0] = single_feq(ws, drho, ux, wsu2x15)
    f1[:, :, 1] = single_feq(ws, drho, uy, wsu2x15)
    f1[:, :, 2] = single_feq(ws, drho, -ux, wsu2x15)
    f1[:, :, 3] = single_feq(ws, drho, -uy, wsu2x15)
    f1[:, :, 4] = single_feq(wd, drho, ux + uy, wdu2x15)
    f1[:, :, 5] = single_feq(wd, drho, uy - ux, wdu2x15)
    f1[:, :, 6] = single_feq(wd, drho, -(ux + uy), wdu2x15)
    f1[:, :, 7] = single_feq(wd, drho, ux - uy, wdu2x15)

    return f0, f1

def stream_and_collide(f0, f1, f2, Nx, Ny, TRT, tauinv, umax , tauinv_minus):
    # direction numbering
    # 6 2 5
    # 3 0 1
    # 7 4 8

    ft0 = f0[:, :]
    ft1 = np.zeros(f1.shape)

    for y in range(0, Ny):
        ym1 = y - 1
        yp1 = y + 1
        for x in range(0, Nx):
            xm1 = x - 1
            xp1 = x + 1

            # Interior
            if x > 0:
                ft1[:, :, 0] = f1[xm1, y, 1]
                if y > 0:
                    ft1[:, :, 4] = f1[xm1, ym1, 5]
                if yp1 < Ny:
                    ft1[:, :, 7] = f1[xm1, yp1, 8]

            # Left wall
            else:
                ft1[:, :, 0] = f1[x, y, 3]
                ft1[:, :, 4] = f1[x, y, 7]
                ft1[:, :, 7] = f1[x, y, 6]

            # Interior
            if xp1 < Nx:
                ft1[:, :, 2] = f1[xp1, y, 3]
                if y > 0:
                    ft1[:, :, 5] = f1[xp1, ym1, 6]
                if yp1 < Ny:
                    ft1[:, :, 6] = f1[xp1, yp1, 7]

            # Right wall
            else:
                ft1[:, :, 0] = f1[x, yp1, 4]

            # Interior
            if y > 0:
                ft1[:, :, 0] = f1[x, ym1, 2]

            # Bottom wall
            else:
                ft1[:, :, 0] = f1[x, y, 4]
                ft1[:, :, 0] = f1[x, y, 4]
                ft1[:, :, 0] = f1[x, y, 4]

            # Interior
            if yp1 < Ny:
                ft1[:, :, 0] = f1[x, yp1, 4]

            # Top wall
            else:
                top_wall_term = 6.0 * wd * umax
                ft1[:, :, 0] = f1[x, y, 2]
                ft1[:, :, 0] = f1[x, y, 5] - top_wall_term
                ft1[:, :, 0] = f1[x, y, 6] + top_wall_term

    # Macroscopic variables
    drho =
