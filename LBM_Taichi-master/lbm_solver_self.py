import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numba.experimental import jitclass


class lbm_solver:
    def __init__(self,
                 nx,
                 ny,
                 niu, # viscosity of fluid
                 bc_type, # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
                 bc_value, # if bc_type = 0, we need to specify the velocity in bc_value
                 cy = 0, # whether to place a cylindrical obstacle
                 cy_para = [0.0, 0.0, 0.0], # location and radius of the cylinder
                 steps = 5000): # total steps to run

        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = np.zeros((nx, ny), dtype=np.float32)
        self.vel = np.zeros((nx, ny, 2), dtype=np.float32)
        self.mask = np.zeros((nx, ny), dtype=np.float32)
        self.f_old = np.zeros((nx, ny, 9), dtype=np.float32)
        self.f_new = np.zeros((nx, ny, 9), dtype=np.float32)
        self.cy = cy
        self.bc_type = np.array(bc_type, dtype=np.int32)
        self.bc_value = np.array(bc_value, dtype=np.float32)
        self.cy_para = np.array(cy_para, dtype=np.float32)
        self.steps = steps
        arr = np.array([ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.w = arr
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e = arr

    def f_eq(self, i, j, k):
        uc = self.e[k, 0] * self.vel[i, j][0] + self.e[k, 1] * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2 + self.vel[i, j][1]**2

        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * uv)

    def initialize(self):
        for j in range(self.rho.shape[1]):
            for i in range(self.rho.shape[0]):
                self.vel[i, j][0] = 0.0
                self.vel[i, j][1] = 0.0
                self.rho[i, j] = 1.0
                self.mask[i, j] = 0.0

                for k in range(9):
                    self.f_new[i, j][k] = self.f_eq(i, j, k)
                    self.f_old[i, j][k] = self.f_new[i, j][k]

                if self.cy == 1:
                    if (i - self.cy_para[0])**2.0 + (j - self.cy_para[1])**2.0 <= self.cy_para[2]**2.0:
                        self.mask[i, j] = 1.0

    def collide_and_stream(self):
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                for k in range(9):
                    ip = i - self.e[k, 0]
                    jp = j - self.e[k, 1]
                    self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] + \
                                            self.f_eq(ip, jp, k) * self.inv_tau
                #####
        # Left wall
        for j in ti.ndrange((0, 1), (1, self.ny-1)):
            self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
            self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i, j][3] + self.f_eq(i, j, 3)*self.inv_tau
            self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j-1][2] + self.f_eq(i, j-1, 2)*self.inv_tau
            self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i+1, j][3] + self.f_eq(i+1, j, 3)*self.inv_tau
            self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j+1][4] + self.f_eq(i, j+1, 4)*self.inv_tau
            self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i, j][7] + self.f_eq(i, j, 7)*self.inv_tau
            self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i+1, j-1][6] + self.f_eq(i+1, j-1, 6)*self.inv_tau
            self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i+1, j+1][7] + self.f_eq(i+1, j+1, 7)*self.inv_tau
            self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i, j][6] + self.f_eq(i, j, 6)*self.inv_tau

        # Right wall
        for i, j in ti.ndrange((self.nx-1, self.nx), (1, self.ny-1)):
            i = self.nx - 1
            self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
            self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i-1, j][1] + self.f_eq(i-1, j, 1)*self.inv_tau
            self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j-1][2] + self.f_eq(i, j-1, 2)*self.inv_tau
            self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i, j][1] + self.f_eq(i, j, 1)*self.inv_tau
            self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j+1][4] + self.f_eq(i, j+1, 4)*self.inv_tau
            self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i-1, j-1][5] + self.f_eq(i-1, j-1, 5)*self.inv_tau
            self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i, j][8] + self.f_eq(i, j, 8)*self.inv_tau
            self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i, j][5] + self.f_eq(i, j, 5)*self.inv_tau
            self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i-1, j+1][8] + self.f_eq(i-1, j+1, 8)*self.inv_tau

        # Bottom wall
        for i, j in ti.ndrange((1, self.nx-1), (0, 1)):
            self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
            self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i-1, j][1] + self.f_eq(i-1, j, 1)*self.inv_tau
            self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j][4] + self.f_eq(i, j, 4)*self.inv_tau
            self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i+1, j][3] + self.f_eq(i+1, j, 3)*self.inv_tau
            self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j+1][4] + self.f_eq(i, j+1, 4)*self.inv_tau
            self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i, j][7] + self.f_eq(i, j, 7)*self.inv_tau
            self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i, j][8] + self.f_eq(i, j, 8)*self.inv_tau
            self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i+1, j+1][7] + self.f_eq(i+1, j+1, 7)*self.inv_tau
            self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i-1, j+1][8] + self.f_eq(i-1, j+1, 8)*self.inv_tau

        # Top wall
        for i, j in ti.ndrange((1, self.nx-1), (self.ny-1, self.ny)):
            self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
            self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i-1, j][1] + self.f_eq(i-1, j, 1)*self.inv_tau
            self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j-1][2] + self.f_eq(i, j-1, 2)*self.inv_tau
            self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i+1, j][3] + self.f_eq(i+1, j, 3)*self.inv_tau
            self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j][2] + self.f_eq(i, j, 2)*self.inv_tau
            self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i-1, j-1][5] + self.f_eq(i-1, j-1, 5)*self.inv_tau
            self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i+1, j-1][6] + self.f_eq(i+1, j-1, 6)*self.inv_tau
            self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i, j][5] + self.f_eq(i, j, 5)*self.inv_tau - \
                                  6 * self.w[5] * self.rho[i, j] * self.bc_value[1, 0]
            self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i, j][6] + self.f_eq(i, j, 6)*self.inv_tau + \
                                  6 * self.w[6] * self.rho[i, j] * self.bc_value[1, 0]

        # Bottom left corner
        i = 0
        j = 0
        self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
        self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i, j][3] + self.f_eq(i, j, 3)*self.inv_tau
        self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j][4] + self.f_eq(i, j, 4)*self.inv_tau
        self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i+1, j][3] + self.f_eq(i+1, j, 3)*self.inv_tau
        self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j+1][4] + self.f_eq(i, j+1, 4)*self.inv_tau
        self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i, j][7] + self.f_eq(i, j, 7)*self.inv_tau
        self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i, j][8] + self.f_eq(i, j, 8)*self.inv_tau
        self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i+1, j+1][7] + self.f_eq(i+1, j+1, 7)*self.inv_tau
        self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i, j][6] + self.f_eq(i, j, 6)*self.inv_tau

        # Bottom right corner
        i = self.nx - 1
        j = 0
        self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
        self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i-1, j][1] + self.f_eq(i-1, j, 1)*self.inv_tau
        self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j][4] + self.f_eq(i, j, 4)*self.inv_tau
        self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i, j][1] + self.f_eq(i, j, 1)*self.inv_tau
        self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j+1][4] + self.f_eq(i, j+1, 4)*self.inv_tau
        self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i, j][7] + self.f_eq(i, j, 7)*self.inv_tau
        self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i, j][8] + self.f_eq(i, j, 8)*self.inv_tau
        self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i, j][5] + self.f_eq(i, j, 5)*self.inv_tau
        self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i-1, j+1][8] + self.f_eq(i-1, j+1, 8)*self.inv_tau

        # Top right corner
        i = self.nx - 1
        j = self.ny - 1
        self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
        self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i-1, j][1] + self.f_eq(i-1, j, 1)*self.inv_tau
        self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j-1][2] + self.f_eq(i, j-1, 2)*self.inv_tau
        self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i, j][1] + self.f_eq(i, j, 1)*self.inv_tau
        self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j][2] + self.f_eq(i, j, 2)*self.inv_tau
        self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i-1, j-1][5] + self.f_eq(i-1, j-1, 5)*self.inv_tau
        self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i, j][8] + self.f_eq(i, j, 8)*self.inv_tau
        self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i, j][5] + self.f_eq(i, j, 5)*self.inv_tau - \
                                  6 * self.w[5] * self.rho[i, j] * self.bc_value[1, 0]
        self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i, j][6] + self.f_eq(i, j, 6)*self.inv_tau + \
                                  6 * self.w[6] * self.rho[i, j] * self.bc_value[1, 0]

        # Top left corner
        i = 0
        j = self.ny - 1
        self.f_new[i, j][0] = (1.0-self.inv_tau)*self.f_old[i, j][0] + self.f_eq(i, j, 0)*self.inv_tau
        self.f_new[i, j][1] = (1.0-self.inv_tau)*self.f_old[i, j][3] + self.f_eq(i, j, 3)*self.inv_tau
        self.f_new[i, j][2] = (1.0-self.inv_tau)*self.f_old[i, j-1][2] + self.f_eq(i, j-1, 2)*self.inv_tau
        self.f_new[i, j][3] = (1.0-self.inv_tau)*self.f_old[i+1, j][3] + self.f_eq(i+1, j, 3)*self.inv_tau
        self.f_new[i, j][4] = (1.0-self.inv_tau)*self.f_old[i, j][2] + self.f_eq(i, j, 2)*self.inv_tau
        self.f_new[i, j][5] = (1.0-self.inv_tau)*self.f_old[i, j][7] + self.f_eq(i, j, 7)*self.inv_tau
        self.f_new[i, j][6] = (1.0-self.inv_tau)*self.f_old[i+1, j-1][6] + self.f_eq(i+1, j-1, 6)*self.inv_tau
        self.f_new[i, j][7] = (1.0-self.inv_tau)*self.f_old[i, j][5] + self.f_eq(i, j, 5)*self.inv_tau - \
                                  6 * self.w[5] * self.rho[i, j] * self.bc_value[1, 0]
        self.f_new[i, j][8] = (1.0-self.inv_tau)*self.f_old[i, j][6] + self.f_eq(i, j, 6)*self.inv_tau + \
                                  6 * self.w[6] * self.rho[i, j] * self.bc_value[1, 0]

    def update_macro_var(self):
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                self.rho[i, j] = 0.0
                self.vel[i, j][0] = 0.0
                self.vel[i, j][1] = 0.0

                for k in range(9):
                    self.f_old[i, j][k] = self.f_new[i, j][k]
                    self.rho[i, j] += self.f_new[i, j][k]
                    self.vel[i, j][0] += self.e[k, 0] * self.f_new[i, j][k]
                    self.vel[i, j][0] += self.e[k, 1] * self.f_new[i, j][k]

                self.vel[i, j][0] /= self.rho[i, j]
                self.vel[i, j][1] /= self.rho[i, j]

    def apply_bc(self):
        for j in range(1, self.ny-1):
            self.apply_bc_core(1, 0, 0, j, 1, j)                    # left
            self.apply_bc_core(1, 2, self.nx-1, j, self.nx-2, j)    # right

        # top and bottom
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny-1, i, self.ny-2)    # top
            self.apply_bc_core(1, 3, i, 0, i, 1)                    # bottom

    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
                self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
        self.rho[ibc, jbc] = self.rho[inb, jnb]

        for k in range(9):
            self.f_old[ibc, jbc][k] = self.f_eq(ibc, jbc, k) - self.f_eq(inb, jnb, k) + \
                                        self.f_old[inb, jnb][k]

    def solve(self):
        self.initialize()

        for i in range(self.steps):
            self.collide_and_stream()
            # print("new step")
            # print(self.f_new[:, :, 0])
            # print(self.f_new[:, :, 1])
            # print(self.f_new[:, :, 2])
            # print(self.f_new[:, :, 3])
            # print(self.f_new[:, :, 4])
            # print(self.f_new[:, :, 5])
            # print(self.f_new[:, :, 6])
            # print(self.f_new[:, :, 7])
            # print(self.f_new[:, :, 8])
            self.update_macro_var()
            self.apply_bc()

    def vel_pass(self):
        return self.vel

Nx = np.int(20)
umax = 0.1

lbm = lbm_solver(Nx, Nx, 0.00796875, [0, 0, 0, 0],
                         [[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]])
lbm.solve()

# Compare results with literature
y_ref, u_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 3))
x_ref, v_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 9))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(lbm.vel_pass()[Nx // 2, :, 0] / umax, np.linspace(0, 1.0, Nx), 'b-', label='LBM')
axes.plot(u_ref, y_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig("ux_Re3200_test_self.png")

plt.clf()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
axes.plot(np.linspace(0, 1.0, Nx), lbm.vel_pass()[:, Nx // 2, 1] / umax, 'b-', label='LBM')
axes.plot(x_ref, v_ref, 'rs', label='Ghia et al. 1982')
axes.legend()
axes.set_xlabel(r'$u_x$')
axes.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig("uy_Re3200_test_self.png")
