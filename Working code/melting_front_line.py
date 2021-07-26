import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import sys


def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr, index=idx, columns=col)
    print(nr, dataset)


np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


path_name = "/Users/Jesper/Documents/MEP/Code/Working code/sim_data/Hsou/octadecane/DT/"
suffix = "_Ra1.e+08_DT0.5_epsilon1e-05_tau0.55_N=100x100_t=2313.0.csv"

fL5 = np.genfromtxt(path_name+"5/fL"+suffix, delimiter=',')
Nx = fL5.shape[1]
Ny = fL5.shape[0]
L = 0.1
x_fL5 = np.zeros((Nx, 1))
y_fL5 = (np.linspace(0, Ny, Ny) + 0.5) / Ny

for j in range(Ny):
    half_fL = np.abs(fL5[j, :] - 0.5)
    idx = half_fL.argmin()
    x_fL5[j] = (idx + 0.5) / Nx

suffix = "_Ra1.e+08_DT0.2_epsilon1e-05_tau0.55_N=100x100_t=2313.0.csv"
fL2 = np.genfromtxt(path_name+"2/fL"+suffix, delimiter=',')
x_fL2 = np.zeros((Nx, 1))
y_fL2 = (np.linspace(0, Ny, Ny) + 0.5) / Ny

for j in range(Ny):
    half_fL = np.abs(fL2[j, :] - 0.5)
    idx = half_fL.argmin()
    x_fL2[j] = (idx + 0.5) / Nx

suffix = "_Ra1.e+08_DT0.1_epsilon1e-05_tau0.55_N=100x100_t=2313.0.csv"
fL1 = np.genfromtxt(path_name+"1/fL"+suffix, delimiter=',')
fL1[77, 8] = 0.5
fL1[92, 16] = 0.5
fL1[93, 17] = 0.5

x_fL1 = np.zeros((Nx, 1))
y_fL1 = (np.linspace(0, Ny, Ny) + 0.5) / Ny

for j in range(Ny):
    half_fL = np.abs(fL1[j, :] - 0.5)
    idx = half_fL.argmin()
    x_fL1[j] = (idx + 0.5) / Nx


suffix = "_Ra1.e+08_DT0.05_epsilon1e-05_tau0.55_N=100x100_t=2313.0.csv"
fL05 = np.genfromtxt(path_name+"0.5/fL"+suffix, delimiter=',')
fL05[93, 17] = 0.5
x_fL05 = np.zeros((Nx, 1))
y_fL05 = (np.linspace(0, Ny, Ny) + 0.5) / Ny

for j in range(Ny):
    half_fL = np.abs(fL05[j, :] - 0.5)
    idx = half_fL.argmin()
    x_fL05[j] = (idx + 0.5) / Nx

fL_bertrand =np.array([[0.033656178529350614, 0.000912514149744581],
                       [0.03996292189340912, 0.021680874165453723],
                       [0.04628699147550071, 0.044337791946773786],
                       [0.048498971977730024, 0.06322144446754596],
                       [0.05071095247995934, 0.08210509698831825],
                       [0.05192090003927276, 0.1028782698423697],
                       [0.054150206759535205, 0.1236504801287529],
                       [0.05553341649917988, 0.16330923063891367],
                       [0.056951278674890846, 0.2067450966802965],
                       [0.059301868921385156, 0.24073721132595616],
                       [0.05970037193614709, 0.28417403993500745],
                       [0.061066255457758664, 0.3219442326795574],
                       [0.06250144385150273, 0.3672686564865508],
                       [0.06412722064361125, 0.4333672157152647],
                       [0.0667550303786356, 0.4975762546106991],
                       [0.06829417608057846, 0.5542320250113583],
                       [0.07078337607133779, 0.6033326017819052],
                       [0.07441321874927807, 0.6656521203440602],
                       [0.08218980294314689, 0.7355220196979847],
                       [0.08771975419872015, 0.7827311509999153],
                       [0.09423441217917618, 0.8261622042029555],
                       [0.10380714764247927, 0.8695903697029901],
                       [0.11324127336151728, 0.8979100730781373],
                       [0.12158673504747382, 0.9186765079585094],
                       [0.1361176565712569, 0.9469913984953142],
                       [0.14541317254602998, 0.9602026397455741],
                       [0.1516506110379559, 0.9734167686988395],
                       [0.15277392750710372, 0.9847471527248365],
                       [0.15394922263035088, 1.0017432100476664]])

plt.plot(fL_bertrand[:, 0], fL_bertrand[:, 1], '--', 'k')

plt.plot(nd.gaussian_filter(x_fL5, 2), y_fL5)
plt.plot(nd.gaussian_filter(x_fL2, 2), y_fL2)
plt.plot(nd.gaussian_filter(x_fL1, 2), y_fL1)
plt.plot(nd.gaussian_filter(x_fL05, 2), y_fL05)
plt.legend([""r"$\Delta T = 0.5$ K", r"$\Delta T = 0.2$ K", r"$\Delta T = 0.1$ K", r"$\Delta T = 0.05$ K"])
plt.xlabel(r'$\bar{x}$', fontsize=18)
plt.ylabel(r'$\bar{y}$', fontsize=18)
plt.grid(alpha=0.3)
plt.xlim(0, 0.5)
plt.ylim(0, 1)
# plt.title(f'Gallium \n Position of melting front, left wall at $T={TH_phys}K$, $t={np.round(t/Nt*Time, decimals=2)}s$')
plt.savefig("/Users/Jesper/Documents/MEP/Code/Working code/Figures/Hsou/octadecane/DT/fL_comparison.png", dpi=300)
