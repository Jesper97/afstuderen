# Fluid solver based on lattice boltzmann method using taichi language
# About taichi : https://github.com/taichi-dev/taichi
# Author : Wang (hietwll@gmail.com)

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import sys

ti.init(arch=ti.gpu)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

@ti.data_oriented
class lbm_solver:
    def __init__(self,
                 nx, # domain size
                 ny,
                 niu, # viscosity of fluid
                 bc_type, # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
                 bc_value, # if bc_type = 0, we need to specify the velocity in bc_value
                 cy = 0, # whether to place a cylindrical obstacle
                 cy_para = [0.0, 0.0, 0.0], # location and radius of the cylinder
                 steps = 5): # total steps to run
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))
        self.cy = cy
        self.cy_para = ti.field(dtype=ti.f32, shape=3)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy_para.from_numpy(np.array(cy_para, dtype=np.float32))
        self.steps = steps
        arr = np.array([ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(arr)

    @ti.func # compute equilibrium distribution function
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(self.e[k, 1],
            ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = 1.0
            self.mask[i, j] = 0.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]
            if(self.cy==1):
                if ((ti.cast(i, ti.f32) - self.cy_para[0])**2.0 + (ti.cast(j, ti.f32)
                    - self.cy_para[1])**2.0 <= self.cy_para[2]**2.0):
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self): # lbm core equation
        #####
        # for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            print(i, j)
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i,j][k] = (1.0-self.inv_tau)*self.f_old[ip,jp][k] + \
                                    self.f_eq(ip,jp,k)*self.inv_tau

        #####
        # Left wall
        for i, j in ti.ndrange((0, 1), (1, self.ny-1)):
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

    @ti.kernel
    def update_macro_var(self): # compute rho u v
        #####
        # for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            self.rho[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0

            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) *
                                      self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) *
                                      self.f_new[i, j][k])

            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self): # impose boundary conditions
        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if (self.cy == 1 and self.mask[i, j] == 1):
                self.vel[i, j][0] = 0.0  # velocity is zero at solid boundary
                self.vel[i, j][1] = 0.0
                inb = 0
                jnb = 0
                if (ti.cast(i,ti.f32) >= self.cy_para[0]):
                    inb = i + 1
                else:
                    inb = i - 1
                if (ti.cast(j,ti.f32) >= self.cy_para[1]):
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if (outer == 1):  # handle outer boundary
            if (self.bc_type[dr] == 0):
                self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):
                self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
                self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc,jbc][k] = self.f_eq(ibc,jbc,k) - self.f_eq(inb,jnb,k) + \
                                        self.f_old[inb,jnb][k]

    def easy_view(self, nr, arr):
        idx = ["idx" for i in arr[1, :]]
        col = ["col" for j in arr[:, 1]]

        dataset = pd.DataFrame(arr.T, index=idx, columns=col)
        print(nr, dataset)

    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.init()
        for i in range(self.steps):
            self.collide_and_stream()

            #####
            # val = self.f_new.to_numpy()
            # print("new step")
            # self.easy_view(0, val[:, :, 0])
            # self.easy_view(1, val[:, :, 1])
            # self.easy_view(2, val[:, :, 2])
            # self.easy_view(3, val[:, :, 3])
            # self.easy_view(4, val[:, :, 4])
            # self.easy_view(5, val[:, :, 5])
            # self.easy_view(6, val[:, :, 6])
            # self.easy_view(7, val[:, :, 7])
            # self.easy_view(8, val[:, :, 8])

            self.update_macro_var()

            #####
            # val = self.vel.to_numpy()
            # self.easy_view("ux", val[:, :, 0])
            # self.easy_view("uy", val[:, :, 1])

            #####
            # self.apply_bc()

            ##  code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0]**2.0+vel[:, :, 1]**2.0)**0.5
            ## color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
                (0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02),cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)
            if (i % 1000 == 0):
                # gui.set_image(img)
                # gui.show()
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')

    def pass_to_py(self):
        return self.vel.to_numpy()[:, :, :]

if __name__ == '__main__':
    flow_case = 1
    Nx = np.int(128)
    umax = 0.05
    if (flow_case == 0):  # von Karman vortex street: Re = U*D/niu = 200
        lbm = lbm_solver(801, 201, 0.01, [0, 0, 1, 0],
             [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
             1,[160.0, 100.0, 20.0])
        lbm.solve()
    elif (flow_case == 1):  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = lbm_solver(Nx, Nx, 0.002, [0, 0, 0, 0],
                         [[0.0, 0.0], [umax, 0.0], [0.0, 0.0], [0.0, 0.0]])
        lbm.solve()

        # Compare results with literature
        y_ref, u_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 3))
        x_ref, v_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 9))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
        axes.plot(lbm.pass_to_py()[Nx // 2, :, 0] / umax, np.linspace(0, 1.0, Nx), 'b-', label='LBM')
        axes.plot(u_ref, y_ref, 'rs', label='Ghia et al. 1982')
        axes.legend()
        axes.set_xlabel(r'$u_x$')
        axes.set_ylabel(r'$y$')
        plt.tight_layout()
        plt.savefig("ux_Re3200_test_bb.png")

        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
        axes.plot(np.linspace(0, 1.0, Nx), lbm.pass_to_py()[:, Nx // 2, 1] / umax, 'b-', label='LBM')
        axes.plot(x_ref, v_ref, 'rs', label='Ghia et al. 1982')
        axes.legend()
        axes.set_xlabel(r'$u_x$')
        axes.set_ylabel(r'$y$')
        plt.tight_layout()
        plt.savefig("uy_Re3200_test_bb.png")
