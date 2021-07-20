import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.special as spec

alpha = 1.4215e-5
Time = 200
xi = 0.21311164

# N = 20
csv_path_tlinN20 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/tlinear/N20/sim_data/"
csv_path_hsouN20 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/hsource/N20/sim_data/"

x_sim_tlinN20 = np.genfromtxt(csv_path_tlinN20+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_sim_hsouN20 = np.genfromtxt(csv_path_hsouN20+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_thN20 = np.genfromtxt(csv_path_hsouN20+"x_th_x_pos_tau0.502_t200.csv", delimiter=',')
tpN20 = np.genfromtxt(csv_path_hsouN20+"tp_x_pos_tau0.502_t200.csv", delimiter=',')
T_hsouN20 = np.genfromtxt(csv_path_hsouN20+"T_x_pos_tau0.502_t200.csv", delimiter=',')
T_tlinN20 = np.genfromtxt(csv_path_tlinN20+"T_x_pos_tau0.502_t200.csv", delimiter=',')
xN20 = np.linspace(0, 0.05, T_hsouN20.shape[0]-2) + 0.5 / 40 * 0.05
T_thN20 = np.zeros(xN20.shape)
for i, pos in enumerate(xN20):
    if pos < x_thN20[-1]:
        T_thN20[i] = 322.5 - (322.5 - 302.8) * spec.erf(pos / (2*np.sqrt(alpha * Time))) / spec.erf(xi)
    else:
        T_thN20[i] = 302.8

# N = 40
csv_path_tlinN40 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/tlinear/N40/sim_data/"
csv_path_hsouN40 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/hsource/N40/sim_data/"

x_sim_tlinN40 = np.genfromtxt(csv_path_tlinN40+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_sim_hsouN40 = np.genfromtxt(csv_path_hsouN40+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_thN40 = np.genfromtxt(csv_path_hsouN40+"x_th_x_pos_tau0.502_t200.csv", delimiter=',')
tpN40 = np.genfromtxt(csv_path_hsouN40+"tp_x_pos_tau0.502_t200.csv", delimiter=',')
T_hsouN40 = np.genfromtxt(csv_path_hsouN40+"T_x_pos_tau0.502_t200.csv", delimiter=',')
T_tlinN40 = np.genfromtxt(csv_path_tlinN40+"T_x_pos_tau0.502_t200.csv", delimiter=',')
xN40 = np.linspace(0, 0.05, T_hsouN40.shape[0]-2) + 0.5 / 40 * 0.05
T_thN40 = np.zeros(xN40.shape)
for i, pos in enumerate(xN40):
    if pos < x_thN40[-1]:
        T_thN40[i] = 322.5 - (322.5 - 302.8) * spec.erf(pos / (2*np.sqrt(alpha * Time))) / spec.erf(xi)
    else:
        T_thN40[i] = 302.8

# N = 80
csv_path_tlinN80 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/tlinear/N80/sim_data/"
csv_path_hsouN80 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/hsource/N80/sim_data/"

x_sim_tlinN80 = np.genfromtxt(csv_path_tlinN80+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_sim_hsouN80 = np.genfromtxt(csv_path_hsouN80+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_thN80 = np.genfromtxt(csv_path_hsouN80+"x_th_x_pos_tau0.502_t200.csv", delimiter=',')
tpN80 = np.genfromtxt(csv_path_hsouN80+"tp_x_pos_tau0.502_t200.csv", delimiter=',')
T_hsouN80 = np.genfromtxt(csv_path_hsouN80+"T_x_pos_tau0.502_t200.csv", delimiter=',')
T_tlinN80 = np.genfromtxt(csv_path_tlinN80+"T_x_pos_tau0.502_t200.csv", delimiter=',')
xN80 = np.linspace(0, 0.05, T_hsouN80.shape[0]-2) + 0.5 / 80 * 0.05
T_thN80 = np.zeros(xN80.shape)
for i, pos in enumerate(xN80):
    if pos < x_thN80[-1]:
        T_thN80[i] = 322.5 - (322.5 - 302.8) * spec.erf(pos / (2*np.sqrt(alpha * Time))) / spec.erf(xi)
    else:
        T_thN80[i] = 302.8

# N = 160
csv_path_tlinN160 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/tlinear/N160/sim_data/"
csv_path_hsouN160 = f"/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/hsource/N160/sim_data/"

x_sim_tlinN160 = np.genfromtxt(csv_path_tlinN160+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_sim_hsouN160 = np.genfromtxt(csv_path_hsouN160+"x_sim_x_pos_tau0.502_t200.csv", delimiter=',')
x_thN160 = np.genfromtxt(csv_path_hsouN160+"x_th_x_pos_tau0.502_t200.csv", delimiter=',')
tpN160 = np.genfromtxt(csv_path_hsouN160+"tp_x_pos_tau0.502_t200.csv", delimiter=',')
T_hsouN160 = np.genfromtxt(csv_path_hsouN160+"T_x_pos_tau0.502_t200.csv", delimiter=',')
T_tlinN160 = np.genfromtxt(csv_path_tlinN160+"T_x_pos_tau0.502_t200.csv", delimiter=',')
xN160 = np.linspace(0, 0.05, T_hsouN160.shape[0]-2) + 0.5 / 160 * 0.05
T_thN160 = np.zeros(xN160.shape)
for i, pos in enumerate(xN160):
    if pos < x_thN160[-1]:
        T_thN160[i] = 322.5 - (322.5 - 302.8) * spec.erf(pos / (2*np.sqrt(alpha * Time))) / spec.erf(xi)
    else:
        T_thN160[i] = 302.8

gs = gridspec.GridSpec(4, 4)

L2_N20_tlin = np.sqrt(np.sum((x_sim_tlinN20 - x_thN20)**2) / x_thN20.shape[0])
L2_N20_hsou = np.sqrt(np.sum((x_sim_hsouN20 - x_thN20)**2) / x_thN20.shape[0])

L2_N40_tlin = np.sqrt(np.sum((x_sim_tlinN40 - x_thN40)**2) / x_thN40.shape[0])
L2_N40_hsou = np.sqrt(np.sum((x_sim_hsouN40 - x_thN40)**2) / x_thN40.shape[0])

L2_N80_tlin = np.sqrt(np.sum((x_sim_tlinN80 - x_thN80)**2) / x_thN80.shape[0])
L2_N80_hsou = np.sqrt(np.sum((x_sim_hsouN80 - x_thN80)**2) / x_thN80.shape[0])

L2_N160_tlin = np.sqrt(np.sum((x_sim_tlinN160 - x_thN160)**2) / x_thN160.shape[0])
L2_N160_hsou = np.sqrt(np.sum((x_sim_hsouN160 - x_thN160)**2) / x_thN160.shape[0])
print(L2_N40_tlin)
print(L2_N40_hsou)
print(L2_N80_tlin)
print(L2_N80_hsou)
print(L2_N160_tlin)
print(L2_N160_hsou)

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(gs[:2, :2])
l1 = ax1.plot(xN40, T_thN40, '--', color='black')
l2 = ax1.plot(xN40, T_hsouN40[1:-1], color="#8b88f8")
l3 = ax1.plot(xN40, T_tlinN40[1:-1], color="#af2f0d")
ax1.grid(alpha=0.2)
# ax1.set_ylim(0, 0.024)
ax1.legend((l1[0], l2[0], l3[0]), (r"Analytical", r"H-source", r"H-linear"))
ax1.set_xlabel(r'$T$ (K)')  #, fontsize=14
ax1.set_ylabel(r'$t$ (s)')  #, fontsize=14
ax1.set_title(f'$N = 40$', fontsize=19)

ax2 = plt.subplot(gs[:2, 2:])
l4 = ax2.plot(xN80, T_thN80, '--', color='black')
l5 = ax2.plot(xN80, T_hsouN80[1:-1], color="#8b88f8")
l6 = ax2.plot(xN80, T_tlinN80[1:-1], color="#af2f0d")
ax2.grid(alpha=0.2)
# ax2.set_ylim(0, 0.024)
ax2.legend((l4[0], l5[0], l6[0]), (r"Analytical", r"H-source", r"H-linear"))
ax2.set_xlabel(r'$T$ (K)')  #, fontsize=14
ax2.set_ylabel(r'$t$ (s)')  #, fontsize=14
ax2.set_title(f'$N = 80$', fontsize=19)

ax3 = plt.subplot(gs[2:4, 1:3])
l7 = ax3.plot(xN160, T_thN160, '--', color='black')
l8 = ax3.plot(xN160, T_hsouN160[1:-1], color="#8b88f8")
l9 = ax3.plot(xN160, T_tlinN160[1:-1], color="#af2f0d")
ax3.grid(alpha=0.2)
# ax3.set_ylim(0, 0.024)
ax3.legend((l7[0], l8[0], l9[0]), (r"Analytical", r"H-source", r"H-linear"))
ax3.set_xlabel(r'$T$ (K)')  #, fontsize=14
ax3.set_ylabel(r'$t$ (s)')  #, fontsize=14
ax3.set_title(f'$N = 160$', fontsize=19)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/" + "T", dpi=300)

gs = gridspec.GridSpec(4, 4)

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(gs[:2, :2])
ax1.plot(tpN40, x_thN40, '--', color='black')
ax1.plot(tpN40, x_sim_hsouN40, color="#f4320c")
ax1.plot(tpN40, x_sim_tlinN40, color="#448ee4")
ax1.grid(alpha=0.2)
ax1.set_ylim(0, 0.024)
ax1.legend([r"Analytical", r"H-source", r"H-linear"])
ax1.set_xlabel(r'$t$ (s)')  #, fontsize=14
ax1.set_ylabel(r'$x$ (m)')  #, fontsize=14
ax1.set_title(f'$N = 40$', fontsize=19)

ax2 = plt.subplot(gs[:2, 2:])
ax2.plot(tpN80, x_thN80, '--', color='black')
ax2.plot(tpN80, x_sim_hsouN80, color="#f4320c")
ax2.plot(tpN80, x_sim_tlinN80, color="#448ee4")
ax2.grid(alpha=0.2)
ax2.set_ylim(0, 0.024)
ax2.legend([r"Analytical", r"H-source", r"H-linear"])
ax2.set_xlabel(r'$t$ (s)')  #, fontsize=14
ax2.set_ylabel(r'$x$ (m)')  #, fontsize=14
ax2.set_title(f'$N = 80$', fontsize=19)

ax3 = plt.subplot(gs[2:4, 1:3])
ax3.plot(tpN160, x_thN160, '--', color='black')
ax3.plot(tpN160, x_sim_hsouN160, color="#f4320c")
ax3.plot(tpN160, x_sim_tlinN160, color="#448ee4")
ax3.grid(alpha=0.2)
ax3.set_ylim(0, 0.024)
ax3.legend([r"Analytical", r"H-source", r"H-linear"])
ax3.set_xlabel(r'$t$ (s)')  #, fontsize=14
ax3.set_ylabel(r'$x$ (m)')  #, fontsize=14
ax3.set_title(f'$N = 160$', fontsize=19)

fig.tight_layout()
fig.savefig("/Users/Jesper/Documents/MEP/Code/Working code/1d_stefan/" + "x_pos", dpi=300)
# plt.show()

