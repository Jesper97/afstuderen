import numpy as np
import matplotlib.pyplot as plt

path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/Tlin/octadecane/Ra108/Nusselt_test/"

Ra = 1e8
FoSte100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
Nu100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/tau=0.55, N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
FoSte100 = np.array(FoSte100[1000:-1:1000])
Nu_corr = (2 * FoSte100)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte100)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte100**(3/2))**(-2))**(-1/2)
Nu100 = 2 * np.array(Nu100[1000:-1:1000])

FoSte140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
Nu140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140_t=11565.0.csv', delimiter=',')
FoSte140 = np.array(FoSte140[1:])
Nu140 = np.array(Nu140[1:])

plt.figure()
plt.plot(FoSte100, Nu100)
plt.plot(FoSte140, Nu140)
plt.plot(FoSte100, Nu_corr, '--')
plt.legend([r"N = 100x100", "N = 140x140", "Jany & Bejan Correlation"])
plt.xlabel('FoSte')
plt.ylabel('Nu')
plt.ylim(0, 70)
# plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_comparison2.png")



