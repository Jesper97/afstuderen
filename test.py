import numpy as np
import pandas as pd
import scipy.interpolate as interp
import sys

# a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# np.savetxt("foo.csv", a, delimiter=",")
#
# x = ".png.csv"
# print(".png.csv".replace(".png", ""))

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


Nx = 300
Ny = 180
Cu = 0.0013333333333333335 / 0.00291917533296844
beta_salt_p = 2.79e-4
T0_p = 920

path1 = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/0degrees/"
path2 = "_freeze_plug_W=0.2_tau=0.52_N=240x192_t=15000.0.csv"
rho = np.genfromtxt(path1+"rho"+path2, delimiter=',')

vel = np.zeros((Nx, Ny, 2))
ux = np.genfromtxt(path1+"ux"+path2, delimiter=',')
uy = np.genfromtxt(path1+"uy"+path2, delimiter=',')
vel[:, :, 0] = ux.T / Cu
vel[:, :, 1] = np.rot90(np.rot90(np.rot90(uy))) / Cu

fL = np.genfromtxt(path1+"fL"+path2, delimiter=',')
fL = fL.T

T = np.zeros((Nx+2, Ny+2))
F = - T[1:-1, 1:-1, None] * rho[:, :, None] * g
T_p = np.genfromtxt(path1+"T"+path2, delimiter=',')
T = beta_salt_p * (T_p.T - T0_p)
