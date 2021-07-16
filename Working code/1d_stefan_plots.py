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

# plt.figure()
# plt.plot(FoSte0515, Nu0515)
# plt.plot(FoSte053, Nu053)
# plt.plot(FoSte055, Nu055)
# plt.plot(FoSte06, Nu06)
# plt.plot(FoSte053, Nu_corr, '--')
# plt.legend([r"$\tau$ = 0.515", r"$\tau$ = 0.53", r"$\tau$ = 0.55", r"$\tau$ = 0.6", "Jany & Bejan Correlation"])
# plt.xlabel('FoSte')
# plt.ylabel('Nu')
# plt.ylim(0, 70)
# # plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
# plt.savefig(path_name + f"Nusselt_correlation_vs_simulation" + "_comparison2.png")

tau = np.array([0.515, 0.53, 0.55, 0.6])
Nu_inf_err = np.array([np.mean(Nu0515[-101:-1]), np.mean(Nu053[-101:-1]), np.mean(Nu055[-101:-1]), np.mean(Nu06[-101:-1])])
plt.plot(tau, Nu_inf_err, 'o-', color='black')
# plt.legend([r"$\tau$ = 0.515", r"$\tau$ = 0.53", r"$\tau$ = 0.55", r"$\tau$ = 0.6", "Jany & Bejan Correlation"])
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$\epsilon$', fontsize=18)
plt.xlim(0.5, 0.61)
# plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
plt.savefig(path_name + f"Nusselt_error" + "tau_comparison.png", dpi=300)
