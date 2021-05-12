import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import sys
import time
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

# Simulation parameters
# Nx                     = 32    # resolution x-dir
# Ny                     = 32     # resolution y-dir
# rho0                   = 1    # average density
# tau                    = 0.6    # collision timescale
# Nt                     = 700   # number of timesteps
plotRealTime = False     # switch on for plotting as the simulation goes along

# # Time = 20
# t_max = 20
# L = 0.01
# rho0 = 6.093e3      # Density (kg/m^3)
# mu = 1.81e-3        # Dynamic viscosity (Ns/m^2)
# nu = mu / rho0      # Kinematic viscosity (m^2/s)  2.97e-7
#
# # Free parameters
# tau = 0.5093
# Nx = 62
# Ny = Nx
# c_s = 1 / np.sqrt(3)
# nu_sim = c_s**2 * (tau - 1/2)
#
# dx = L / (Nx-2)
# dt = c_s**2 * (tau - 1/2) * (dx**2 / nu)
#
# Nt = np.int(t_max / dt)
# print(Nt)

### Free parameters
Re_lbm      = 3200.0
u_lbm       = 0.05
L_lbm       = 102
t_max       = 20.0

# Deduce other parameters
cs          = 1.0/math.sqrt(3.0)
Nx          = L_lbm - 2
Ny = Nx
nu_sim      = u_lbm*(L_lbm-2)/Re_lbm
tau_plus    = 0.5 + nu_sim/(cs**2)
tau = tau_plus
rho0     = 1.0
dt          = Re_lbm*nu_sim/L_lbm**2
Nt      = math.floor(t_max/dt)

print("tau", tau_plus)
print("Nt", Nt)

Lambda = 1/4
tau_minus = Lambda / (tau_plus - 1/2) + 1/2

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
copp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])  # sums to 1

# Collision operators
M_rho = np.ones(9)
M_e = np.array([-4, -1, -1, -1, -1, 2, 2, 2, 2])
M_eps = np.array([4, -2, -2, -2, -2, 1, 1, 1, 1])
M_jx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
M_qx = np.array([0, -2, 0, 2, 0, 1, -1, -1, 1])
M_jy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
M_qy = np.array([0, 0, -2, 0, 2, 1, 1, -1, -1])
M_pxx = np.array([0, 1, -1, 1, -1, 0, 0, 0, 0])
M_pyy = np.array([0, 0, 0, 0, 0, 1, -1, 1, -1])
M = np.array([M_rho, M_e, M_eps, M_jx, M_qx, M_jy, M_qy, M_pxx, M_pyy])
M_inv = np.dot(M.T, np.linalg.inv(np.dot(M, M.T)))


s0 = 0
s1 = 1.64
s2 = 1.2
s3 = 0
s7 = 1 / tau_plus
s4 = 8 * ((2 - s7) / (8 - s7))
s5 = 0
s6 = s4
s8 = s7
S = np.diag(np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8]))

uxw = u_lbm
uyw = 0

Re2 = uxw * Nx / nu_sim
print("Re", Re2)

@njit
def feq_func(rho_, u_x, u_y, idxs, cxs, cys, weights, c_opp):
    feq = np.zeros((Ny, Nx, NL))
    # feq_minus = np.zeros((Ny, Nx, NL))
    # feq_plus = np.zeros((Ny, Nx, NL))

    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        feq[:, :, i] = rho_ * w * (1 + 3 * (cx * u_x + cy * u_y) + 9 * (cx * u_x + cy * u_y) ** 2 / 2 - 3 * (u_x ** 2 + u_y ** 2) / 2)

    # feq_plus[:, :, 0] = feq[:, :, 0]
    # feq_plus[:, :, 1] = (feq[:, :, 1] + feq[:, :, 3]) / 2
    # feq_plus[:, :, 2] = (feq[:, :, 2] + feq[:, :, 4]) / 2
    # feq_plus[:, :, 3] = (feq[:, :, 3] + feq[:, :, 1]) / 2
    # feq_plus[:, :, 4] = (feq[:, :, 4] + feq[:, :, 2]) / 2
    # feq_plus[:, :, 5] = (feq[:, :, 5] + feq[:, :, 7]) / 2
    # feq_plus[:, :, 6] = (feq[:, :, 6] + feq[:, :, 8]) / 2
    # feq_plus[:, :, 7] = (feq[:, :, 7] + feq[:, :, 5]) / 2
    # feq_plus[:, :, 8] = (feq[:, :, 8] + feq[:, :, 6]) / 2
    # feq_minus[:, :, 0] = 0
    # feq_minus[:, :, 1] = (feq[:, :, 1] - feq[:, :, 3]) / 2
    # feq_minus[:, :, 2] = (feq[:, :, 2] - feq[:, :, 4]) / 2
    # feq_minus[:, :, 3] = (feq[:, :, 3] - feq[:, :, 1]) / 2
    # feq_minus[:, :, 4] = (feq[:, :, 4] - feq[:, :, 2]) / 2
    # feq_minus[:, :, 5] = (feq[:, :, 5] - feq[:, :, 7]) / 2
    # feq_minus[:, :, 6] = (feq[:, :, 6] - feq[:, :, 8]) / 2
    # feq_minus[:, :, 7] = (feq[:, :, 7] - feq[:, :, 5]) / 2
    # feq_minus[:, :, 8] = (feq[:, :, 8] - feq[:, :, 6]) / 2

    return feq#, feq_minus, feq_plus

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)

# @njit
def streaming(f, idxs, cxs, cys, c_opp):
    # f_plus = np.zeros(f.shape)
    # f_minus = np.zeros(f.shape)

    for i, cx, cy, copp in zip(idxs, cxs, cys, c_opp):
        f[:, :, i] = np.roll(f[:, :, i], cx, axis=0)
        f[:, :, i] = np.roll(f[:, :, i], cy, axis=1)

    # f_plus[:, :, 0] = f[:, :, 0]
    # f_plus[:, :, 1] = (f[:, :, 1] + f[:, :, 3]) / 2
    # f_plus[:, :, 2] = (f[:, :, 2] + f[:, :, 4]) / 2
    # f_plus[:, :, 3] = (f[:, :, 3] + f[:, :, 1]) / 2
    # f_plus[:, :, 4] = (f[:, :, 4] + f[:, :, 2]) / 2
    # f_plus[:, :, 5] = (f[:, :, 5] + f[:, :, 7]) / 2
    # f_plus[:, :, 6] = (f[:, :, 6] + f[:, :, 8]) / 2
    # f_plus[:, :, 7] = (f[:, :, 7] + f[:, :, 5]) / 2
    # f_plus[:, :, 8] = (f[:, :, 8] + f[:, :, 6]) / 2
    # f_minus[:, :, 0] = 0
    # f_minus[:, :, 1] = (f[:, :, 1] - f[:, :, 3]) / 2
    # f_minus[:, :, 2] = (f[:, :, 2] - f[:, :, 4]) / 2
    # f_minus[:, :, 3] = (f[:, :, 3] - f[:, :, 1]) / 2
    # f_minus[:, :, 4] = (f[:, :, 4] - f[:, :, 2]) / 2
    # f_minus[:, :, 5] = (f[:, :, 5] - f[:, :, 7]) / 2
    # f_minus[:, :, 6] = (f[:, :, 6] - f[:, :, 8]) / 2
    # f_minus[:, :, 7] = (f[:, :, 7] - f[:, :, 5]) / 2
    # f_minus[:, :, 8] = (f[:, :, 8] - f[:, :, 6]) / 2

    return f#, f_minus, f_plus


# Initial Conditions
# f = np.ones((Ny, Nx, NL)) #* rho0 / NL
# np.random.seed(42)
# f += 0.01*np.random.randn(Ny, Nx, NL)
# X, Y = np.meshgrid(range(Nx), range(Ny))
# f[:, :, 3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))

rho = np.ones((Nx, Ny)) * rho0
ux = np.zeros((Nx, Ny))
uy = np.zeros((Nx, Ny))
# f, f_minus, f_plus = feq_func(rho, ux, uy, idxs, cxs, cys, weights, copp)
f = feq_func(rho, ux, uy, idxs, cxs, cys, weights, copp)

# for i in idxs:
#     f[:, :, i] *= rho0 / rho

# Cylinder boundary
walls = np.zeros((Nx, Ny), dtype=bool)
walls[:, :] = False
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True

# Prep figure
fig = plt.figure(figsize=(4, 2), dpi=80)

# Simulation Main Loop
Nt2 = 3
for it in range(Nt2):
    if it % 10000 == 0:
        print(it)

    # Drift
    # f, f_minus, f_plus = streaming(f, idxs, cxs, cys, copp)
    f = streaming(f, idxs, cxs, cys, copp)

    # for i, cx, cy in zip(idxs, cxs, cys):
    #     f[:, :, i] = np.roll(f[:, :, i], cx, axis=0)
    #     f[:, :, i] = np.roll(f[:, :, i], cy, axis=1)

    # Set reflective boundaries
    bndryF = f[walls, :]
    bndryF = bndryF[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]

    # Calculate fluid variables
    rho = np.sum(f, 2)
    ux = (f[:, :, 1] + f[:, :, 5] + f[:, :, 8] - (f[:, :, 3] + f[:, :, 6] + f[:, :, 7])) / rho
    uy = (f[:, :, 2] + f[:, :, 5] + f[:, :, 6] - (f[:, :, 4] + f[:, :, 7] + f[:, :, 8])) / rho
    # ux = np.sum(f * cxs, 2) / rho
    # uy = np.sum(f * cys, 2) / rho

    # Apply Collision
    # feq, feq_minus, feq_plus = feq_func(rho, ux, uy, idxs, cxs, cys, weights, copp)
    feq = feq_func(rho, ux, uy, idxs, cxs, cys, weights, copp)

    ### Collision
    Mf = np.einsum('ij,klj->kli', M, f - feq)
    SMf = np.einsum('ij,klj->kli', S, Mf)
    MSMf = np.einsum('ij,klj->kli', M_inv, SMf)

    f += f - MSMf
    # f += f - 1/tau_plus * (f_plus - feq_plus) - 1/tau_minus * (f_minus - feq_minus)
    # f += -(1.0 / tau) * (f - feq)

    # Apply boundary
    f[walls, :] = bndryF

    # # Apply moving wall
    # f[:, -1, 7] += - 2 * weights[5] * rho[:, -2] * (cxs[5] * uxw + cys[5] * uyw) * 3
    # f[:, -1, 4] += - 2 * weights[2] * rho[:, -2] * (cxs[2] * uxw + cys[2] * uyw) * 3
    # f[:, -1, 8] += - 2 * weights[6] * rho[:, -2] * (cxs[6] * uxw + cys[6] * uyw) * 3

    easy_view(it, f[:, :, 1])

    # plot in real time - color 1/2 particles blue, other half red
    if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
        plt.cla()
        ux[walls] = 0
        uy[walls] = 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[walls] = np.nan
        cmap = plt.cm.bwr
        cmap.set_bad('black')
        plt.imshow(vorticity, cmap='bwr')
        plt.clim(-.1, .1)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.pause(0.001)

y = np.linspace(0, Ny-1, Ny)
# easy_view(it, rho[:, :])

# Save figure
# plt.savefig("Figures/Princeton/pois_test_vort2", dpi=240)
# plt.show()

# Save figure
plt.figure()
plt.title('Arrowplot velocity')
plt.xlabel('x')
plt.ylabel('y')
plt.quiver(ux[1:-1, 1:-1].T, uy[1:-1, 1:-1].T)
# plt.show()
plt.savefig("Figures/Princeton/arrow_test")
#
# uxt = ux.T
# # print(uxt)
# plt.figure()
# plt.title('Lineplot $u_x$')
# plt.xlabel('$u_x$')
# plt.ylabel('$y$')
# plt.plot(uxt[1:-1, np.int(Nx/2)]/uxw, y[1:-1])
# plt.savefig("Figures/Princeton/pois_test_line4")


ux_aslan_Re200 = np.array([[-0.0025942920610690923, 0.002395112976734781],
                            [-0.04149336741922438, 0.05010708719721124],
                            [-0.0700109310639741, 0.09158308073992039],
                            [-0.09853118352294643, 0.13098868733123825],
                            [-0.15816639416568679, 0.21187649242116346],
                            [-0.1970574030811742, 0.2657996274958143],
                            [-0.22556690028325616, 0.31348678189269774],
                            [-0.24368681932926495, 0.36114911646598813],
                            [-0.24361959897370028, 0.4129087902507734],
                            [-0.20716465574387588, 0.4832150772155054],
                            [-0.1057466164892566, 0.5751053032723903],
                            [0.032056456825399704, 0.6834717555578307],
                            [0.13346508523024014, 0.7681156272848457],
                            [0.253062197443765, 0.8578920316990513],
                            [0.3102357986657277, 0.8815649726102904],
                            [0.4011822509304849, 0.9103332164733304],
                            [0.5102997097114799, 0.930776477839517],
                            [0.5908350733063834, 0.9430064459150189],
                            [0.7311239553698208, 0.9654456347618174],
                            [0.8532189759961282, 0.9786115170186426],
                            [0.9571308904421962, 0.99078564049106]])

uy_aslan_Re200 = np.array([[0.0028410198132926556, 0.005643065347558263],
                            [0.01674252975891327, 0.04011458438033971],
                            [0.03366044377344865, 0.06884467929296556],
                            [0.05257014857405415, 0.10044799908102753],
                            [0.11638129973066716, 0.16654509426647332],
                            [0.20427240149250972, 0.20465947762158437],
                            [0.2742558476803124, 0.21045331189064235],
                            [0.324282411728386, 0.201155895064830675],
                            [0.3833479315939864, 0.1782239421047342],
                            [0.41039710687772946, 0.16101259279468144],
                            [0.45148933027829685, 0.12873440259609728],
                            [0.48357336702899056, 0.0993215398532516],
                            [0.5477824660872488, 0.026136869462831835],
                            [0.612036693258065, -0.06284264032278908],
                            [0.6592458271880466, -0.1360395158163778],
                            [0.7205205732911315, -0.2322006518960988],
                            [0.75966818529603, -0.28386485361055674],
                            [0.7837378795120418, -0.3082578292147094],
                            [0.8097870568470628, -0.32546989647200725],
                            [0.8237972845183906, -0.32904958143675594],
                            [0.8437787819922421, -0.32257369728472346],
                            [0.8637336128541274, -0.3067644989446176],
                            [0.881663832484446, -0.282341369556165],
                            [0.9154709426237074, -0.2148299182976035]])

ux_aslan_Re1000 = np.array([[-0.04084507042253527, 0.014034257574583764],
                            [-0.08873239436619718, 0.032835933394330885],
                            [-0.1507042253521127, 0.05790204285023348],
                            [-0.19999999999999996, 0.07982756057677998],
                            [-0.2549295774647888, 0.10592378411991188],
                            [-0.3056338028169014, 0.1309731437532583],
                            [-0.3450704225352113, 0.1570463362903749],
                            [-0.3563380281690141, 0.17891532336579308],
                            [-0.32816901408450705, 0.14349363611429244],
                            [-0.3507042253521127, 0.16641995573859392],
                            [-0.3591549295774648, 0.18932533808469976],
                            [-0.3507042253521127, 0.21220559569697062],
                            [-0.33380281690140845, 0.23299212548967052],
                            [-0.30845070422535215, 0.25376609291545327],
                            [-0.29014084507042254, 0.27246936352768036],
                            [-0.2732394366197182, 0.28701239696241976],
                            [-0.25352112676056343, 0.3036324083941736],
                            [-0.22394366197183102, 0.3348059218997649],
                            [-0.19577464788732396, 0.36598152913317583],
                            [-0.16760563380281696, 0.39715713636658667],
                            [-0.14084507042253525, 0.4262536718751637],
                            [-0.05211267605633807, 0.5156118814866305],
                            [0.015492957746478853, 0.5800275115835493],
                            [0.06619718309859168, 0.6278189427930749],
                            [0.1507042253521126, 0.7046964408720795],
                            [0.2239436619718309, 0.7649413651524131],
                            [0.30845070422535237, 0.8418188632314177],
                            [0.33098591549295775, 0.8667593490181464],
                            [0.3549295774647889, 0.910428230150937],
                            [0.37605633802816896, 0.9332896442128317],
                            [0.4169014084507041, 0.9488376670009653],
                            [0.46056338028168997, 0.9581380059754993],
                            [0.5647887323943661, 0.9694294801064453],
                            [0.6239436619718308, 0.974544457169657],
                            [0.6887323943661972, 0.9806916420479171],
                            [0.840845070422535, 0.9887901812540174],
                            [0.9732394366197183, 0.9979586153759185]])

uy_aslan_Re1000 = np.array([[0.006005803547594141, 0.06358922252653859],
                            [0.056414311742717965, 0.21235005304433607],
                            [0.09815950920245389, 0.2811282051282053],
                            [0.13907307840639874, 0.3231661425661389],
                            [0.15950920245398767, 0.33528205128205146],
                            [0.18200408997955012, 0.339794871794872],
                            [0.2085889570552147, 0.33189743589743614],
                            [0.23514384607157787, 0.3122478357661508],
                            [0.33380650053960503, 0.21161919863558853],
                            [0.4304241652325812, 0.11026254301056354],
                            [0.5224049533476579, 0.008851601497595896],
                            [0.604802485100094, -0.08542113625899539],
                            [0.6969493678680575, -0.18936951514199762],
                            [0.7868373633567647, -0.2890030932123121],
                            [0.8519942701395843, -0.3980552701099276],
                            [0.8701431492842534, -0.4386666666666662],
                            [0.8946830265848668, -0.488307692307692],
                            [0.9089979550102246, -0.49846153846153796],
                            [0.9182004089979549, -0.48605128205128156],
                            [0.9282074919551971, -0.46390016409592116],
                            [0.9582669764939612, -0.2907308935100663],
                            [0.9716712456766355, -0.18788595574081524],
                            [0.9871830833001144, -0.0694432731828194]])

uy_lin_Re3200 = np.array([[0.00259291270527226, 0.062058371735791096],
                            [0.013828867761452035, 0.17726574500768055],
                            [0.03457216940363009, 0.3078341013824885],
                            [0.061365600691443395, 0.39231950844854074],
                            [0.07865168539325842, 0.41689708141321047],
                            [0.09334485738980122, 0.4337941628264209],
                            [0.11754537597234227, 0.41996927803379425],
                            [0.1495246326707001, 0.38463901689708146],
                            [0.17891097666378564, 0.346236559139785],
                            [0.21175453759723423, 0.3093701996927804],
                            [0.26793431287813313, 0.2494623655913979],
                            [0.38980121002592916, 0.12350230414746544],
                            [0.5280898876404495, -0.014746543778801802],
                            [0.6490924805531547, -0.13917050691244237],
                            [0.7856525496974935, -0.2897081413210445],
                            [0.9023336214347452, -0.43717357910906285],
                            [0.9222126188418324, -0.4894009216589862],
                            [0.9377700950734659, -0.5447004608294931],
                            [0.948141745894555, -0.5631336405529953],
                            [0.958513396715644, -0.535483870967742],
                            [0.9688850475367329, -0.4463901689708142],
                            [0.9853068280034573, -0.1913978494623656],
                            [0.9982713915298186, -0.02857142857142858]])

y = np.linspace(1/2, Nx-1/2, Nx-2)
x = np.linspace(1/2, Nx-1/2, Nx-2)

# plt.figure()
# plt.title('$u_x$ at $x=L/2$ for Re=1000')
# plt.xlabel('$u_x/u_0$')
# plt.ylabel('$y/L$')
# plt.plot(ux[np.int(Nx/2), 1:-1]/uxw, y/Nx)
# plt.plot(ux_aslan_Re1000[:, 0], ux_aslan_Re1000[:, 1], 'o')
# plt.savefig(f"Figures/Princeton/ux_Re1000.png")

plt.figure()
plt.title(f'$u_x$ at $x=L/2$ for Re={Re_lbm}')
plt.xlabel('$u_x/u_0$')
plt.ylabel('$y/L$')
plt.plot(ux[np.int(Nx/2), 1:-1]/uxw, y/Nx)
# plt.plot(ux_lin_Re3200[:, 0], ux_lin_Re3200[:, 1], 'o')
plt.savefig(f"Figures/Princeton/ux_test.png")

# plt.figure()
# plt.title('$u_y$ at $y=L/2$ for Re=1000')
# plt.xlabel('$u_y/u_0$')
# plt.ylabel('$x/L$')
# plt.plot(x/Nx, uy[1:-1, np.int(Nx/2)]/uxw)
# plt.plot(uy_aslan_Re1000[:, 0], uy_aslan_Re1000[:, 1], 'o')
# plt.savefig(f"Figures/Princeton/uy_Re1000.png")

plt.figure()
plt.title(f'$u_y$ at $x=L/2$ for Re={Re_lbm}')
plt.xlabel('$u_y/u_0$')
plt.ylabel('$x/L$')
plt.plot(x/Nx, uy[1:-1, np.int(Nx/2)]/uxw)
plt.plot(uy_lin_Re3200[:, 0], uy_lin_Re3200[:, 1], 'o')
plt.savefig(f"Figures/Princeton/uy_test.png")

