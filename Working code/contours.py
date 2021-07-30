import matplotlib.pyplot as plt
import numpy as np

# csv_path_Ra4 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra104/sim_data/"
# csv_path_Ra5 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra105/sim_data/"
# csv_path_Ra6 = f"/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra106/sim_data/"
#
# u4 = np.genfromtxt(csv_path_Ra4+"_u_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')
# w4 = np.genfromtxt(csv_path_Ra4+"_w_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')
# T4 = np.genfromtxt(csv_path_Ra4+"_T_naturalconv_Ra9999_tau=0.55_N=100x100.csv", delimiter=',')
#
# u5 = np.genfromtxt(csv_path_Ra5+"_u_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')
# w5 = np.genfromtxt(csv_path_Ra5+"_w_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')
# T5 = np.genfromtxt(csv_path_Ra5+"_T_naturalconv_Ra99999_tau=0.55_N=100x100.csv", delimiter=',')
#
# u6 = np.genfromtxt(csv_path_Ra6+"_u_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
# w6 = np.genfromtxt(csv_path_Ra6+"_w_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
# T6 = np.genfromtxt(csv_path_Ra6+"_T_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
#
# x = np.linspace(0, 1, 100)
# y = np.linspace(0, 1, 100)
# u = np.linspace(0, 1, 100)
# X, Y = np.meshgrid(x, y)
#
# # plt.figure()
# # plt.clf()
# # CS = plt.contour(X, Y, u4)
# # plt.clabel(CS, inline=True)
# # plt.xlabel(r'$\bar{x}$')
# # plt.ylabel(r'$\bar{y}$')
# # plt.title(f'Ra $=10^4$')
# # plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_u.png")
#
# u4 = np.flip(np.rot90(u4), axis=0)
# u5 = np.flip(np.rot90(u5), axis=0)
# u6 = np.flip(np.rot90(u6), axis=0)
#
# w4 = np.flip(np.rot90(w4), axis=0)
# w5 = np.flip(np.rot90(w5), axis=0)
# w6 = np.flip(np.rot90(w6), axis=0)
#
# T4 = T4[1:-1, 1:-1].T
# T5 = T5[1:-1, 1:-1].T
# T6 = T6[1:-1, 1:-1].T
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# ax1.contour(X, Y, u4, np.linspace(u4.min(), u4.max(), 11), origin="lower")
# # ax1.clabel(CS, inline=True)
# ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax1.set_title(f'Ra $=10^4$', fontsize=16)
#
# ax2.contour(X, Y, u5, np.linspace(u5.min(), u5.max(), 11), origin="lower")
# # ax2.clabel(CS, inline=True)
# ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax2.set_title(f'Ra $=10^5$', fontsize=16)
#
# ax3.contour(X, Y, u6, np.linspace(u6.min(), u6.max(), 11), origin="upper")
# # ax3.clabel(CS, inline=True)
# ax3.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax3.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax3.set_title(f'Ra $=10^6$', fontsize=16)
#
# fig.tight_layout()
# fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_u.png")
# fig.clf()
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# ax1.contour(X, Y, w4, np.linspace(w4.min(), w4.max(), 11), origin="lower")
# # ax1.clabel(CS, inline=True)
# ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax1.set_title(f'Ra $=10^4$', fontsize=16)
#
# ax2.contour(X, Y, w5, np.linspace(w5.min(), w5.max(), 11), origin="lower")
# # ax2.clabel(CS, inline=True)
# ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax2.set_title(f'Ra $=10^5$', fontsize=16)
#
# ax3.contour(X, Y, w6, np.linspace(w6.min(), w6.max(), 11), origin="upper")
# # ax3.clabel(CS, inline=True)
# ax3.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax3.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax3.set_title(f'Ra $=10^6$', fontsize=16)
#
# fig.tight_layout()
# fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_w.png")
# fig.clf()
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# ax1.contour(X, Y, T4, np.linspace(T4.min(), T4.max(), 11))
# # ax1.clabel(CS, inline=True)
# ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax1.set_title(f'Ra $=10^4$', fontsize=16)
#
# ax2.contour(X, Y, T5, np.linspace(T5.min(), T5.max(), 11))
# # ax2.clabel(CS, inline=True)
# ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax2.set_title(f'Ra $=10^5$', fontsize=16)
#
# ax3.contour(X, Y, T6, np.linspace(T6.min(), T6.max(), 11))
# # ax3.clabel(CS, inline=True)
# ax3.set_xlabel(r'$\bar{x}$', fontsize=13)
# ax3.set_ylabel(r'$\bar{y}$', fontsize=13)
# ax3.set_title(f'Ra $=10^6$', fontsize=16)
#
# fig.tight_layout()
# fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"contour_T.png")

# #### Convergence
# csv_path_Nu = "/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/Ra106/sim_data/"
# Nu40 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=40x40.csv", delimiter=',')
# Nu60 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=60x60.csv", delimiter=',')
# Nu80 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=80x80.csv", delimiter=',')
# Nu100 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=100x100.csv", delimiter=',')
# Nu120 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=120x120.csv", delimiter=',')
# Nu140 = np.genfromtxt(csv_path_Nu+"_Nu_naturalconv_Ra1000000_tau=0.55_N=140x140.csv", delimiter=',')
#
# Nu = np.array([Nu40, Nu60, Nu80, Nu100, Nu120, Nu140])
# mesh_size = np.array([40, 60, 80, 100, 120, 140])
#
# plt.figure()
# plt.plot(mesh_size, Nu, '-o', color="#388004", mfc='k', mec='k')
# plt.grid(alpha=0.2)
# plt.xlabel(r'Number of nodes per side', fontsize=14)
# plt.ylabel(r'Nu$_0$ ', fontsize=14)
# plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/natural_convection/" + f"convergence_Nu.png")

##### Octadecane
N = 240
path_hsou = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N240/"
path_hlin = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N240/"

ux_hsou = np.genfromtxt(path_hsou+"ux_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')
uy_hsou = np.genfromtxt(path_hsou+"uy_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')
T_hsou = np.genfromtxt(path_hsou+"T_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')

ux_hlin = np.genfromtxt(path_hlin+"ux_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')
uy_hlin = np.genfromtxt(path_hlin+"uy_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')
T_hlin = np.genfromtxt(path_hlin+"T_Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240_t=11565.0.csv", delimiter=',')

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
u = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# ux_hsou = np.flip(np.rot90(ux_hsou), axis=0)
# ux_hlin = np.flip(np.rot90(ux_hlin), axis=0)
#
uy_hsou = np.flip(uy_hsou, axis=0)
uy_hlin = np.flip(uy_hlin, axis=0)

T_hsou = T_hsou
T_hlin = T_hlin

u_hsou = np.sqrt(ux_hsou**2 + uy_hsou**2)
u_hlin = np.sqrt(ux_hlin**2 + uy_hlin**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.contour(X, Y, ux_hsou, np.linspace(ux_hsou.min(), ux_hsou.max(), 15), origin="lower")
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
ax1.set_title(f'H-source', fontsize=16)

ax2.contour(X, Y, ux_hlin, np.linspace(ux_hlin.min(), ux_hlin.max(), 15), origin="lower")
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
ax2.set_title(f'H-linear', fontsize=16)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/" + f"contour_ux_oct.png", dpi=300)
fig.clf()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(u_hsou, cmap='RdBu_r', interpolation='spline16', origin='lower', extent=[0, 1, 0, 1])
# ax1.contour(X, Y, u_hsou, np.linspace(u_hsou.min(), u_hsou.max(), 15), cmap='RdBu_r', origin="lower")
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
ax1.set_title(f'H-source', fontsize=16)

im = ax2.imshow(u_hlin, cmap='RdBu_r', interpolation='spline16', origin='lower', extent=[0, 1, 0, 1])
# ax2.contour(X, Y, u_hlin, np.linspace(u_hlin.min(), u_hlin.max(), 15), cmap='RdBu_r', origin="lower")
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
ax2.set_title(f'H-linear', fontsize=16)

# fig.subplots_adjust(wspace=0.3, hspace=0.0)
# cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(im, cax=cb_ax)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/" + f"contour_u_oct.png", dpi=300)
fig.clf()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.contour(X, Y, uy_hsou, np.linspace(uy_hsou.min(), uy_hsou.max(), 15), origin="lower")
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
ax1.set_title(f'H-source', fontsize=16)

ax2.contour(X, Y, uy_hlin, np.linspace(uy_hsou.min(), uy_hsou.max(), 15), origin="lower")
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
ax2.set_title(f'H-linear', fontsize=16)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/" + f"contour_uy_oct.png", dpi=300)
fig.clf()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.contour(X, Y, T_hsou, np.linspace(T_hsou.min(), T_hsou.max(), 11))
# ax1.clabel(CS, inline=True)
ax1.set_xlabel(r'$\bar{x}$', fontsize=13)
ax1.set_ylabel(r'$\bar{y}$', fontsize=13)
ax1.set_title(f'H-source', fontsize=16)

ax2.contour(X, Y, T_hlin, np.linspace(T_hsou.min(), T_hsou.max(), 11))
# ax2.clabel(CS, inline=True)
ax2.set_xlabel(r'$\bar{x}$', fontsize=13)
ax2.set_ylabel(r'$\bar{y}$', fontsize=13)
ax2.set_title(f'H-linear', fontsize=16)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/" + f"contour_T_oct.png", dpi=300)


