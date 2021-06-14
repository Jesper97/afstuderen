import numpy as np
import matplotlib.pyplot as plt

path_name = "/Users/Jesper/Documents/MEP/Code/Working code/Figures/Tlin/octadecane/Ra108/Nusselt_test/"

Ra = 1e8
FoSte140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140.png.csv', delimiter=',')
Nu140 = np.genfromtxt('/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Tlin/octadecane/Ra108/N140/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=140x140.png.csv', delimiter=',')

FoSte100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/N100/FoSteRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')
Nu100 = np.genfromtxt('/Users/Jesper/Documents/MEP/Large files/N100/NuRa1.e+08_Pr50.0_Ste0.1_tau0.55_N=100x100.png.csv', delimiter=',')

FoSte140 = np.array(FoSte140[1:])
Nu_corr140 = (2 * FoSte140)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte140)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte140**(3/2))**(-2))**(-1/2)
Nu140 = 2*np.array(Nu140[1:])
mse_140 = np.sqrt(((Nu_corr140 - Nu140)**2).mean())
print(Nu140[-10])
print(mse_140)

FoSte100 = np.array(FoSte100[1:])
Nu_corr100 = (2 * FoSte100)**(-1/2) + (0.35 * Ra**(1/4) - (2 * FoSte100)**(-1/2)) * (1 + (0.0175 * Ra**(3/4) * FoSte100**(3/2))**(-2))**(-1/2)
Nu100 = 2*np.array(Nu100[1:])

mse_100 = np.sqrt(((Nu_corr100 - Nu100)**2).mean())
print(Nu100[-10])
print(mse_100)

p = np.log((mse_100-mse_140) / mse_140) / np.log(140/100)
print(p)
