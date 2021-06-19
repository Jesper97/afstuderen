import numpy as np
import matplotlib.pyplot as plt

path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/Tlin/octadecane/Ra108/Nusselt_test/"

Ra = 1e8
FoSte053 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.53/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.53_N=100x100.csv', delimiter=',')
Nu053 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.53/NuRa1.e+08_Pr50.0_Ste0.1_tau0.53_N=100x100.csv', delimiter=',')
FoSte053 = np.array(FoSte053[1:])
Nu_corr = (2 * FoSte053)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte053)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte053**(3/2))**(-2))**(-1/2)
Nu053 = np.array(Nu053[1:])

FoSte055 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100_t=11565.0.csv', delimiter=',')
Nu055 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100_t=11565.0.csv', delimiter=',')
FoSte055 = np.array(FoSte055[1:])
Nu055 = np.array(Nu055[1:])

FoSte06 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.6/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.6_N=100x100.csv', delimiter=',')
Nu06 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.6/NuRa1.e+08_Pr50.0_Ste0.1_tau0.6_N=100x100.csv', delimiter=',')
FoSte06 = np.array(FoSte06[1:])
Nu06 = np.array(Nu06[1:])

FoSte0515 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.515/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.515_N=100x100.csv', delimiter=',')
Nu0515 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/Ra108/N100/tau0.515/NuRa1.e+08_Pr50.0_Ste0.1_tau0.515_N=100x100.csv', delimiter=',')
FoSte0515 = np.array(FoSte0515[1:])
Nu0515 = np.array(Nu0515[1:])

plt.figure()
plt.plot(FoSte0515, Nu0515)
plt.plot(FoSte053, Nu053)
plt.plot(FoSte055, Nu055)
plt.plot(FoSte06, Nu06)
plt.plot(FoSte053, Nu_corr, '--')
plt.legend([r"$\tau$ = 0.515", r"$\tau$ = 0.53", r"$\tau$ = 0.55", r"$\tau$ = 0.6", "Jany & Bejan Correlation"])
plt.xlabel('FoSte')
plt.ylabel('Nu')
plt.ylim(0, 70)
# plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_comparison2.png")


# Ra = 1e8
# FoSte100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
# Nu100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
# FoSte100 = np.array(FoSte100[1:])
# Nu_corr = (2 * FoSte100)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte100)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte100**(3/2))**(-2))**(-1/2)
# Nu100 = 2 * np.array(Nu100[1:])
#
# FoSte140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
# Nu140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
# FoSte140 = np.array(FoSte140[1:])
# Nu140 = np.array(Nu140[1:])
#
# FoSte196 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N196/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=196x196_t=11565.0.csv', delimiter=',')
# Nu196 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N196/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=196x196_t=11565.0.csv', delimiter=',')
# FoSte196 = np.array(FoSte196[1:])
# Nu196 = np.array(Nu196[1:])
#
# plt.figure()
# plt.plot(FoSte100, Nu100)
# plt.plot(FoSte140, Nu140)
# plt.plot(FoSte196, Nu196)
# plt.plot(FoSte100, Nu_corr, '--')
# plt.legend([r"$N$ = 100x100", r"$N$ = 140x140", r"$N$ = 196x196", "Jany & Bejan Correlation"])
# plt.xlabel('FoSte')
# plt.ylabel('Nu')
# plt.ylim(0, 70)
# # # plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_grid_convergence.png")
