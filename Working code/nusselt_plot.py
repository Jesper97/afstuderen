import numpy as np
import matplotlib.pyplot as plt

path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/"

# Ra = 1e8
# FoSte0525 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.525/FoSteRa1.e+08_tau0.525_N=100x100_t11565.csv', delimiter=',')
# Nu0525 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.525/NuRa1.e+08_tau0.525_N=100x100_t11565.csv', delimiter=',')
# FoSte0525 = np.array(FoSte0525[1:])
# Nu_corr = (2 * FoSte0525)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte0525)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte0525**(3/2))**(-2))**(-1/2)
# Nu0525 = np.array(Nu0525[1:])
#
# FoSte055 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100_t=11565.0.csv', delimiter=',')
# Nu055 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100_t=11565.0.csv', delimiter=',')
# FoSte055 = np.array(FoSte055[1:])
# Nu055 = np.array(Nu055[1:])
#
# FoSte06 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.6/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.6_N=100x100.csv', delimiter=',')
# Nu06 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.6/NuRa1.e+08_Pr50.0_Ste0.1_tau0.6_N=100x100.csv', delimiter=',')
# FoSte06 = np.array(FoSte06[1:])
# Nu06 = np.array(Nu06[1:])
#
# FoSte0515 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.515/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.515_N=100x100.csv', delimiter=',')
# Nu0515 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.515/NuRa1.e+08_Pr50.0_Ste0.1_tau0.515_N=100x100.csv', delimiter=',')
# FoSte0515 = np.array(FoSte0515[1:])
# Nu0515 = np.array(Nu0515[1:])
#
# plt.figure()
# plt.plot(10*FoSte0515, Nu0515)
# plt.plot(10*FoSte0525, Nu0525)
# plt.plot(10*FoSte055, Nu055)
# plt.plot(10*FoSte06, Nu06)
# plt.plot(10*FoSte0525, Nu_corr, '--', color='k')
# plt.legend([r"$\tau$ = 0.515", r"$\tau$ = 0.525", r"$\tau$ = 0.55", r"$\tau$ = 0.6", "Jany & Bejan"])
# plt.xlabel('Fo', fontsize=14)
# plt.ylabel(r'Nu$_0$', fontsize=14)
# plt.ylim(0, 70)
# plt.xlim(right=0.1)
# plt.grid(alpha=0.3)
# plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_comparison2.png", dpi=300)

# tau = np.array([0.515, 0.53, 0.55, 0.6])
# Nu_inf_err = np.array([np.mean(Nu0515[-101:-1]), np.mean(Nu053[-101:-1]), np.mean(Nu055[-101:-1]), np.mean(Nu06[-101:-1])])
# plt.plot(tau, Nu_inf_err, 'o-', color='black')
# # plt.legend([r"$\tau$ = 0.515", r"$\tau$ = 0.53", r"$\tau$ = 0.55", r"$\tau$ = 0.6", "Jany & Bejan Correlation"])
# plt.xlabel(r'$\tau$', fontsize=18)
# plt.ylabel(r'$\epsilon$', fontsize=18)
# plt.xlim(0.5, 0.61)
# # plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.savefig(path_name + f"Nusselt_error" + "tau_comparison.png", dpi=300)


Ra = 1e8
FoSte100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
Nu100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
FoSte100 = np.array(FoSte100[1:])
Nu_corr = (2 * FoSte100)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte100)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte100**(3/2))**(-2))**(-1/2)
Nu100 = 2 * np.array(Nu100[1:])

FoSte140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
Nu140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
FoSte140 = np.array(FoSte140[1:])
Nu140 = np.array(Nu140[1:])

FoSte196 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N196/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=196x196_t=11565.0.csv', delimiter=',')
Nu196 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N196/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=196x196_t=11565.0.csv', delimiter=',')
FoSte196 = np.array(FoSte196[1:])
Nu196 = np.array(Nu196[1:])

FoSte240 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N240/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
Nu240 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N240/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
FoSte240 = np.array(FoSte240[1:])
Nu240 = np.array(Nu240[1:])

Nu100_source = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.55/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.csv', delimiter=',')
Nu100_source = np.array(Nu100_source[1:])

Nu140_source = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N140/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140.csv', delimiter=',')
Nu140_source = np.array(Nu140_source[1:])

Nu196_source = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N196/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=196x196.csv', delimiter=',')
Nu196_source = np.array(Nu196_source[1:])

Nu240_source = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N240/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
Nu240_source = np.array(Nu240_source[1:])
#
# print("Hlin")
# print(Nu100[-1])
# print(Nu140[-1])
# print(Nu196[-1])
# print(Nu240[-1])
#
# print("Hsou")
# print(Nu100_source[-1])
# print(Nu140_source[-1])
# print(Nu196_source[-1])
# print(Nu240_source[-1])
#
# plt.figure()
# plt.plot(10*FoSte100, Nu100)
# plt.plot(10*FoSte140, Nu140)
# plt.plot(10*FoSte196, Nu196)
# plt.plot(10*FoSte240, Nu240)
# plt.plot(10*FoSte100, Nu_corr, '--')
# plt.legend([r"$N$ = 100x100", r"$N$ = 140x140", r"$N$ = 196x196", r"$N$ = 240x240", "Jany & Bejan Correlation"])
# plt.xlabel('Fo')
# plt.ylabel('Nu')
# plt.ylim(0, 70)
# # plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_grid_convergence.png", dpi=300)
# plt.clf()
#
errors = np.array([Nu100[-1], Nu140[-1], Nu196[-1], Nu240[-1]]) - 35
N = np.array([100, 140, 196, 240])

x = np.linspace(100, 240, 50)
order2 = 2e5 * x**-2
order3 = 1.5e7 * x**-3

print(errors)
plt.clf()
plt.figure()
plt.loglog(N, errors, '-o', color="#029386", mfc='k', mec='k', markersize=5)
plt.loglog(x, order2, '--', color='k')
plt.loglog(x, order3, ':', color='k')
plt.legend([r"Error in Nu$_0$", r"$\sim N^{-2}$", r"$\sim N^{-3}$"], fontsize=14, loc=1)
plt.ylabel(r'Error', fontsize=14)
plt.xlabel(r'$N$', fontsize=14)

import matplotlib.ticker as mticker
ax = plt.gca()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.get_minor_formatter().set_scientific(False)
# ax.invert_yaxis()
# ax.set_ylim(ax.get_ylim()[::-1])
plt.grid(True, which="both", alpha=0.3)
plt.xlim(100, 240)
plt.ylim(22, 1)
plt.savefig(path_name+"grid_convergence.png", dpi=300)


# FoSte240 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N240/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
# Nu240 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N240/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
# FoSte240 = np.array(FoSte240[1:])
# Nu240 = np.array(Nu240[1:])
#
# Nu240_source = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N240/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=240x240.csv', delimiter=',')
# Nu240_source = np.array(Nu240_source[1:])
#
# Nu_corr = (2 * FoSte240)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte240)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte240**(3/2))**(-2))**(-1/2)
#
# plt.figure()
# plt.plot(10*FoSte240, Nu_corr, color="k", linestyle='--')
# plt.plot(10*FoSte240, Nu240, color="#2c6fbb")
# plt.plot(10*FoSte240, Nu240_source, color="#8c000f")
# plt.legend(["Jany & Bejan Correlation", r"H-Linear", r"H-source"])
# plt.xlabel('Fo', fontsize=13)
# plt.ylabel(r'Nu$_0$', fontsize=13)
# plt.ylim(0, 70)
# plt.xlim(0, 0.1)
# plt.grid(alpha=0.3)
# plt.savefig(path_name + f"Nu_240_comparion", dpi=300)
# plt.clf()
