import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import sys
from matplotlib import cm
from scipy.ndimage.interpolation import map_coordinates

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


path1 = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/0degrees/L=0.4/fL_test/N=300/"
path2 = "_freeze_plug_0deg_tau=0.502_N=300x180_test_t=1240.0.csv"

W_wall = 0.02
L_cooled = 0.2
W = 0.2 + 2 * W_wall
L = 0.4

beta_salt_p = 2.79e-4
T0_p = 846

Nx = 300
Ny = 180
Nx_new = 320
Ny_new = np.int(W / L * Nx_new)
print("Nodes:", Nx_new, "x", Ny_new)

rho = np.genfromtxt(path1+"rho"+path2, delimiter=',')
ux = np.genfromtxt(path1+"ux"+path2, delimiter=',')
uy = np.genfromtxt(path1+"uy"+path2, delimiter=',')
fL = np.genfromtxt(path1+"fL"+path2, delimiter=',')
fL = fL.T
T_p = np.genfromtxt(path1+"T"+path2, delimiter=',')
T_origin = beta_salt_p * (T_p.T - T0_p)

vel = np.zeros((Nx, Ny, 2))
vel[:, :, 0] = ux.T
vel[:, :, 1] = np.rot90(np.rot90(np.rot90(uy)))

print(rho.shape)
print(vel[:, :, 0].shape)
print(vel[:, :, 1].shape)

new_dims = []
for original_len, new_len in zip(T_origin.shape, (Nx_new, Ny_new)):
    new_dims.append(np.linspace(0, original_len-1, new_len))

coords = np.meshgrid(*new_dims, indexing='ij')
fL_new = map_coordinates(fL, coords, order=1)
rho_new = map_coordinates(rho, coords, order=2)
ux_new = map_coordinates(vel[:, :, 0], coords, order=2)
uy_new = map_coordinates(vel[:, :, 1], coords, order=2)

dim = (Nx_new, Ny_new)
idx_boundary = np.int(W_wall / W * Ny_new)
idx_cooled = np.int(L_cooled / L * Nx_new)

# Retouch liquid fraction
for j in range(idx_boundary+1, Ny_new-idx_boundary+1):
    for i in range(Nx_new-120, Nx_new+1):
        fL_new[i-1, j-1] = 1

# Temperature has bigger domain
new_dims = []
for original_len, new_len in zip(T_origin.shape, (Nx_new+2, Ny_new+2)):
    new_dims.append(np.linspace(0, original_len-1, new_len))

coords = np.meshgrid(*new_dims, indexing='ij')
T_new = map_coordinates(T_origin, coords, order=1)


# View field
cmap = cm.get_cmap('PiYG', 11)
plt.figure()
plt.clf()
# plt.imshow(T_new[1:-1, 1:-1].T, origin='lower', cmap=cmap)
# plt.imshow(fL_new.T, cmap=cm.RdBu, origin='lower', aspect=1.0)
plt.imshow(np.rot90(uy_new), cmap=cm.Blues)
# plt.show()
plt.imshow(ux_new.T, cmap=cm.Blues, origin='lower')
plt.xlabel('$x$ (# lattice nodes)')
plt.ylabel('$y$ (# lattice nodes)')
plt.title(f'LiF-ThF$_4$ \n $T$')
plt.colorbar()
# plt.show()

# Save file location
csv_path = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Freeze Plug/0degrees/L=0.4/fL_test/Interpolated/N320/"
csv_file = "_freeze_plug_0deg_tau=0.502_N=320x192_interp_t=1240.0.csv"

# Streamlines velocity
Cu = 0.001333333 / 0.000583835066593688
ux_phys = vel[:, :, 0] * Cu
uy_phys = vel[:, :, 1] * Cu
uy_plot = np.rot90(uy_phys)
ux_plot = ux_phys.T
T_phys = T_new / beta_salt_p + T0_p

# Save arrays to CSV-files
np.savetxt(csv_path+"rho_"+csv_file+f"_t=1240.csv",    rho_new,       delimiter=",")
np.savetxt(csv_path+"fL_"+csv_file+f"_t=1240.csv",     fL.T,      delimiter=",")
np.savetxt(csv_path+"ux_"+csv_file+f"_t=1240.csv",     ux_plot,   delimiter=",")
np.savetxt(csv_path+"uy_"+csv_file+f"_t=1240.csv",     uy_plot,   delimiter=",")
np.savetxt(csv_path+"T_"+csv_file+f"_t=1240.csv",      T_phys.T,  delimiter=",")
