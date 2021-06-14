import numpy as np
import matplotlib.pyplot as plt
import csv

path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/Tlin/octadecane/Ra108/Nusselt_test/"

Ra = 1e8
FoSte = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=120x120.png.csv', delimiter=',')
Nu = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=120x120.png.csv', delimiter=',')

FoSte = np.array(FoSte[1:])
Nu_corr = (2 * FoSte)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte**(3/2))**(-2))**(-1/2)
Nu = np.array(Nu[1:])

plt.figure()
plt.plot(FoSte, Nu)
plt.plot(FoSte, Nu_corr)
plt.legend([r"$\tau=0.55$", "Jany & Bejan Correlation"])
plt.xlabel('FoSte')
plt.ylabel('Nu')
plt.ylim(0, 80)
# plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "Ra1.e+08_Pr50.0_Ste0.1_tau0.55_N=120x120.png")
