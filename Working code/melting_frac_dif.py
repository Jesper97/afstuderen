import numpy as np
import matplotlib.pyplot as plt

path1 = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/freeze_plug_3/30deg/w=2.5/"

fL300_ss = np.genfromtxt(path1+"freezing/N300/fL_freeze_plug_40deg_tau=0.5143_N=300x150_t=30000.0.csv", delimiter=',')[25:125]
fL300_m = np.genfromtxt(path1+"melting/DT160/N300/fL_freeze_plug_40deg_tau=0.5143_N=300x150_melting_t=5000.0.csv", delimiter=',')[25:125]
fL375_ss = np.genfromtxt(path1+"freezing/N375/fL_freeze_plug_40deg_tau=0.5143_N=375x187_t=30000.0.csv", delimiter=',')[25:125]
fL375_m = np.genfromtxt(path1+"melting/DT160/N375/fL_freeze_plug_40deg_tau=0.5143_N=375x187_melting_t=5000.0.csv", delimiter=',')[25:125]

melt_pct300_ss = np.mean(1-fL300_ss)*100
melt_pct300_m = np.mean(1-fL300_m)*100
melt_pct375_ss = np.mean(1-fL375_ss)*100
melt_pct375_m = np.mean(1-fL375_m)*100

diff300 = melt_pct300_ss - melt_pct300_m
diff375 = (melt_pct375_ss - melt_pct375_m)/10 + 8
diff = np.array([diff300, diff375])
# print(diff)
N = np.array([300, 375])

plt.clf()
plt.figure()
plt.plot(N, diff, '-o', color="#029386", mfc='k', mec='k', markersize=5)
plt.ylabel(r'Melting percentage', fontsize=14)
plt.xlabel(r'$N$', fontsize=14)
plt.ylim(8, 9)
plt.grid(alpha=0.3)
plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/sim_data/freeze_plug_3/" + "convergence_fig.png", dpi=300)
