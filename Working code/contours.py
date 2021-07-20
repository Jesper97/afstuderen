import matplotlib.pyplot as plt
import numpy as np

csv_path_Ra4 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra104/sim_data/"
csv_path_Ra5 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra105/sim_data/"
csv_path_Ra6 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra106/sim_data/"

u4 = np.genfromtxt(csv_path_Ra4+"_u_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')
w4 = np.genfromtxt(csv_path_Ra4+"_w_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')
T4 = np.genfromtxt(csv_path_Ra4+"_T_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')

u5 = np.genfromtxt(csv_path_Ra5+"_u_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')
w5 = np.genfromtxt(csv_path_Ra5+"_w_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')
T5 = np.genfromtxt(csv_path_Ra5+"_T_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')

u6 = np.genfromtxt(csv_path_Ra6+"_u_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
w6 = np.genfromtxt(csv_path_Ra6+"_w_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
T6 = np.genfromtxt(csv_path_Ra6+"_T_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
u = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# plt.figure()
# plt.clf()
# CS = plt.contour(X, Y, u4)
# plt.clabel(CS, inline=True)
# plt.xlabel(r'$\bar{x}$')
# plt.ylabel(r'$\bar{y}$')
# plt.title(f'Ra $=10^4$')
# plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_u.png")

u4 = np.flip(np.rot90(u4), axis=0)
u5 = np.flip(np.rot90(u5), axis=0)
u6 = np.flip(np.rot90(u6), axis=0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.contour(X, Y, u4, origin="lower")
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$')
ax1.set_ylabel(r'$\bar{y}$')
ax1.set_title(f'Ra $=10^4$')

ax2.contour(X, Y, u5, origin="lower")
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$')
ax2.set_ylabel(r'$\bar{y}$')
ax2.set_title(f'Ra $=10^5$')

ax3.contour(X, Y, u6, origin="upper")
# ax3.clabel(CS, inline=True)
ax3.set_xlabel(r'$\bar{x}$')
ax3.set_ylabel(r'$\bar{y}$')
ax3.set_title(f'Ra $=10^6$')

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_u.png")
fig.clf()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.contour(X, Y, T4[1:-1, 1:-1])
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$')
ax1.set_ylabel(r'$\bar{y}$')
ax1.set_title(f'Ra $=10^4$')

ax2.contour(X, Y, T5[1:-1, 1:-1])
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$')
ax2.set_ylabel(r'$\bar{y}$')
ax2.set_title(f'Ra $=10^5$')

ax3.contour(X, Y, T6[1:-1, 1:-1])
# ax3.clabel(CS, inline=True)
ax3.set_xlabel(r'$\bar{x}$')
ax3.set_ylabel(r'$\bar{y}$')
ax3.set_title(f'Ra $=10^6$')

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_T.png")

