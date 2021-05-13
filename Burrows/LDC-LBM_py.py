import csv
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

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
f1 = np.zeros((Nx, Ny, q-1), dtype=float)
f2 = np.zeros((Nx, Ny, q-1), dtype=float)
rho = np.ones((Nx, Ny), dtype=float)
ux = np.ones((Nx, Ny), dtype=float)
uy = np.ones((Nx, Ny), dtype=float)

w0 = 4.0/9.0
ws = 1.0/9.0
wd = 1.0/9.0

# Field indices?

# Equilibrium distribution
@njit
def single_feq(w, drho, cdotu, wu2x15):
    return w * (drho + (3.0 * cdotu) + (4.5 * (cdotu * cdotu))) - wu2x15


@njit
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


@njit
def stream_and_collide(f0, f1, fnew, Nx, Ny, TRT, tauinv, umax, tauinv_minus):
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
                ft1[x, y, 0] = f1[xm1, y, 0]
                if y > 0:
                    ft1[x, y, 4] = f1[xm1, ym1, 4]
                if yp1 < Ny:
                    ft1[x, y, 7] = f1[xm1, yp1, 7]

            # Left wall
            else:
                ft1[x, y, 0] = f1[x, y, 2]
                ft1[x, y, 4] = f1[x, y, 6]
                ft1[x, y, 7] = f1[x, y, 5]

            # Interior
            if xp1 < Nx:
                ft1[x, y, 2] = f1[xp1, y, 2]
                if y > 0:
                    ft1[x, y, 5] = f1[xp1, ym1, 5]
                if yp1 < Ny:
                    ft1[x, y, 6] = f1[xp1, yp1, 6]

            # Right wall
            else:
                ft1[x, y, 2] = f1[x, y, 0]
                ft1[x, y, 5] = f1[x, y, 7]
                ft1[x, y, 6] = f1[x, y, 5]

            # Interior
            if y > 0:
                ft1[x, y, 1] = f1[x, ym1, 1]

            # Bottom wall
            else:
                ft1[x, y, 1] = f1[x, y, 3]
                ft1[x, y, 4] = f1[x, y, 6]
                ft1[x, y, 5] = f1[x, y, 7]

            # Interior
            if yp1 < Ny:
                ft1[x, y, 3] = f1[x, yp1, 3]

            # Top wall
            else:
                top_wall_term = 6.0 * wd * umax
                ft1[x, y, 3] = f1[x, y, 2]
                ft1[x, y, 6] = f1[x, y, 5] - top_wall_term
                ft1[x, y, 7] = f1[x, y, 6] + top_wall_term

    # Macroscopic variables
    drho = ft0 + ft1[:, :, 0] + ft1[:, :, 1] + ft1[:, :, 2] + ft1[:, :, 3] + ft1[:, :, 4] + ft1[:, :, 5] + ft1[:, :, 6]
    ux = ft1[:, :, 0] + ft1[:, :, 4] + ft1[:, :, 7] - (ft1[:, :, 2] + ft1[:, :, 5] + ft1[:, :, 6])
    uy = ft1[:, :, 1] + ft1[:, :, 4] + ft1[:, :, 5] - (ft1[:, :, 3] + ft1[:, :, 6] + ft1[:, :, 7])

    fnew = np.zeros((Nx, Ny, 9))

    # Collision
    # TRT
    # w * (drho + (3.0 * cdotu) + (4.5 * (cdotu * cdotu))) - wu2x15
    if TRT == 1:
        half_tauinv = 0.5 * tauinv
        half_tauinv_minus = 0.5 * tauinv_minus
        u2x15 = 1.5 * ((ux * ux) + (uy * uy))
        wsu2x15 = ws * u2x15
        wdu2x15 = wd * u2x15
        wsdrho = ws * drho
        wddrho = wd * drho

        # k = 0
        fnew[:, :, 0] = ft0 - tauinv * (ft0 - w0 * (drho - u2x15))

        # k = 1, 3
        cdotu = ux
        feq_term1 = wsdrho + ws * (4.5 * (cdotu * cdotu)) - wsu2x15
        feq_term2 = 3.0 * ws * cdotu
        fplus = half_tauinv * ((ft1[:, :, 0] + ft1[:, :, 2]) - 2 * feq_term1)
        fminus = half_tauinv_minus * ((ft1[:, :, 0] - ft1[:, :, 2]) - 2 * feq_term2)
        fnew[:, :, 1] = ft1[:, :, 0] - fplus - fminus
        fnew[:, :, 3] = ft1[:, :, 2] - fplus + fminus

        # k = 2, 4
        cdotu = uy
        feq_term1 = wsdrho + ws * (4.5 * (cdotu * cdotu)) - wsu2x15
        feq_term2 = 3.0 * ws * cdotu
        fplus = half_tauinv * ((ft1[:, :, 1] + ft1[:, :, 3]) - 2 * feq_term1)
        fminus = half_tauinv_minus * ((ft1[:, :, 1] - ft1[:, :, 3]) - 2 * feq_term2)
        fnew[:, :, 2] = ft1[:, :, 1] - fplus - fminus
        fnew[:, :, 4] = ft1[:, :, 3] - fplus + fminus

        # k = 5, 7
        cdotu = ux + uy
        feq_term1 = wddrho + wd * (4.5 * (cdotu * cdotu)) - wdu2x15
        feq_term2 = 3.0 * wd * cdotu
        fplus = half_tauinv * ((ft1[:, :, 4] + ft1[:, :, 6]) - 2 * feq_term1)
        fminus = half_tauinv_minus * ((ft1[:, :, 4] - ft1[:, :, 6]) - 2 * feq_term2)
        fnew[:, :, 5] = ft1[:, :, 4] - fplus - fminus
        fnew[:, :, 7] = ft1[:, :, 6] - fplus + fminus

        # k = 6, 8
        cdotu = -ux + uy
        feq_term1 = wddrho + wd * (4.5 * (cdotu * cdotu)) - wdu2x15
        feq_term2 = 3.0 * wd * cdotu
        fplus = half_tauinv * ((ft1[:, :, 5] + ft1[:, :, 7]) - 2 * feq_term1)
        fminus = half_tauinv_minus * ((ft1[:, :, 5] - ft1[:, :, 7]) - 2 * feq_term2)
        fnew[:, :, 6] = ft1[:, :, 5] - fplus - fminus
        fnew[:, :, 8] = ft1[:, :, 7] - fplus + fminus

    # SRT/BGK
    else:
        u2x15 = 1.5 * ((ux * ux) + (uy * uy))
        wsu2x15 = ws * u2x15
        wdu2x15 = wd * u2x15
        omtauinv = 1.0 - tauinv

        fnew[:, :, 0] = omtauinv * ft0 + tauinv * w0 * (drho - u2x15)
        fnew[:, :, 1] = omtauinv * ft1[:, :, 0] + tauinv * single_feq(ws, drho, ux, wsu2x15)
        fnew[:, :, 2] = omtauinv * ft1[:, :, 1] + tauinv * single_feq(ws, drho, uy, wsu2x15)
        fnew[:, :, 3] = omtauinv * ft1[:, :, 2] + tauinv * single_feq(ws, drho, -ux, wsu2x15)
        fnew[:, :, 4] = omtauinv * ft1[:, :, 3] + tauinv * single_feq(ws, drho, -uy, wsu2x15)
        fnew[:, :, 5] = omtauinv * ft1[:, :, 4] + tauinv * single_feq(wd, drho, ux + uy, wdu2x15)
        fnew[:, :, 6] = omtauinv * ft1[:, :, 5] + tauinv * single_feq(wd, drho, -ux + uy, wdu2x15)
        fnew[:, :, 7] = omtauinv * ft1[:, :, 6] + tauinv * single_feq(wd, drho, -ux - uy, wdu2x15)
        fnew[:, :, 8] = omtauinv * ft1[:, :, 7] + tauinv * single_feq(wd, drho, ux - uy, wdu2x15)

    f0 = fnew[:, :, 0]
    f2 = fnew[:, :, 1:-1]

    return f0, f1, f2

# Compute macro variables
def macros(f0, f2):
    r = 1.0 + f0 + f2[:, :, 0] + f2[:, :, 1] + f2[:, :, 2] + f2[:, :, 3] + f2[:, :, 4] + f2[:, :, 5] + f2[:, :, 6] + f2[:, :, 7]
    u = f2[:, :, 0] + f2[:, :, 4] + f2[:, :, 7] - (f2[:, :, 2] + f2[:, :, 5] + f2[:, :, 6])
    v = f2[:, :, 1] + f2[:, :, 4] + f2[:, :, 5] - (f2[:, :, 3] + f2[:, :, 6] + f2[:, :, 7])

    return r, u, v

# Initialize f at equilibrium
f0, f1 = eq_dist_func(w0, ws, wd, f0, f1, rho, ux, uy)

for t in range(0, int(Nsteps)):
    msg = (t % Nmsg == 0)
    if t % 2 == 0:
        stream_and_collide(f0, f1, f2, Nx, Ny, TRT, omega, umax_sim, omega_minus)
    else:
        stream_and_collide(f0, f2, f1, Nx, Ny, TRT, omega, umax_sim, omega_minus)
    # if msg:

