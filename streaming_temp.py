from numba import njit

@njit
def fluid_temp(Nx, Ny, g_plus, g_minus, g_star):
    for i in range(1, Nx-1):
        g_plus[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
        g_plus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] + g_star[i+1, 1:Ny-1, 3]) / 2
        g_plus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] + g_star[i, 2:Ny, 4]) / 2
        g_plus[i, 1:Ny-1, 3] = g_plus[i, 1:Ny-1, 1]
        g_plus[i, 1:Ny-1, 4] = g_plus[i, 1:Ny-1, 2]

        g_minus[i, 1:Ny-1, 0] = 0
        g_minus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] - g_star[i+1, 1:Ny-1, 3]) / 2
        g_minus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] - g_star[i, 2:Ny, 4]) / 2
        g_minus[i, 1:Ny-1, 3] = -g_minus[i, 1:Ny-1, 1]
        g_minus[i, 1:Ny-1, 4] = -g_minus[i, 1:Ny-1, 2]

    return g_plus, g_minus

@njit
def left_wall_temp(Ny, g_plus, g_minus, g_star, w, T):
    i = 0

    g_plus[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
    g_plus[i, 1:Ny-1, 1] = (-g_star[i, 1:Ny-1, 3] + 2 * w[3] * T[1:Ny-1] + g_star[i+1, 1:Ny-1, 3]) / 2      # Bounce
    g_plus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] + g_star[i, 2:Ny, 4]) / 2
    g_plus[i, 1:Ny-1, 3] = g_plus[i, 1:Ny-1, 1]
    g_plus[i, 1:Ny-1, 4] = g_plus[i, 1:Ny-1, 2]                                 # Bounce

    g_minus[i, 1:Ny-1, 0] = 0
    g_minus[i, 1:Ny-1, 1] = (-g_star[i, 1:Ny-1, 3] + 2 * w[3] * T[1:Ny-1] - g_star[i+1, 1:Ny-1, 3]) / 2     # Bounce
    g_minus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] - g_star[i, 2:Ny, 4]) / 2
    g_minus[i, 1:Ny-1, 3] = -g_minus[i, 1:Ny-1, 1]
    g_minus[i, 1:Ny-1, 4] = -g_minus[i, 1:Ny-1, 2]

    return g_plus, g_minus

@njit
def right_wall_temp(Nx, Ny, g_plus, g_minus, g_star, w, T):
    i = Nx - 1

    g = g_plus.copy()

    g[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
    g[i, 1:Ny-1, 1] = g_star[i-1, 1:Ny-1, 1]      # Bounce
    g[i, 1:Ny-1, 2] = g_star[i, 0:Ny-2, 2]
    g[i, 1:Ny-1, 3] = -g_star[i, 1:Ny-1, 1] + 2 * w[1] * T[1:Ny-1]
    g[i, 1:Ny-1, 4] = g_star[i, 2:Ny, 2]

    # g_plus[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
    # g_plus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] - g_star[i, 1:Ny-1, 1] + 2 * w[1] * T[1:Ny-1]) / 2      # Bounce
    # g_plus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] + g_star[i, 2:Ny, 4]) / 2
    # g_plus[i, 1:Ny-1, 3] = g_plus[i, 1:Ny-1, 1]
    # g_plus[i, 1:Ny-1, 4] = g_plus[i, 1:Ny-1, 2]
    #
    # g_minus[i, 1:Ny-1, 0] = 0
    # g_minus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] + g_star[i, 1:Ny-1, 1] - 2 * w[1] * T[1:Ny-1]) / 2     # Bounce
    # g_minus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] - g_star[i, 2:Ny, 4]) / 2
    # g_minus[i, 1:Ny-1, 3] = -g_minus[i, 1:Ny-1, 1]
    # g_minus[i, 1:Ny-1, 4] = -g_minus[i, 1:Ny-1, 2]

    g_plus[i, 1:Ny-1, 0] = g[i, 1:Ny-1, 0]
    g_plus[i, 1:Ny-1, 1] = (g[i, 1:Ny-1, 1] + g[i, 1:Ny-1, 3]) / 2
    g_plus[i, 1:Ny-1, 2] = (g[i, 1:Ny-1, 2] + g[i, 1:Ny-1, 4]) / 2
    g_plus[i, 1:Ny-1, 3] = (g[i, 1:Ny-1, 3] + g[i, 1:Ny-1, 1]) / 2
    g_plus[i, 1:Ny-1, 4] = (g[i, 1:Ny-1, 4] + g[i, 1:Ny-1, 2]) / 2

    g_minus[i, 1:Ny-1, 0] = 0
    g_minus[i, 1:Ny-1, 1] = (g[i, 1:Ny-1, 1] - g[i, 1:Ny-1, 3]) / 2
    g_minus[i, 1:Ny-1, 2] = (g[i, 1:Ny-1, 2] - g[i, 1:Ny-1, 4]) / 2
    g_minus[i, 1:Ny-1, 3] = (g[i, 1:Ny-1, 3] - g[i, 1:Ny-1, 1]) / 2
    g_minus[i, 1:Ny-1, 4] = (g[i, 1:Ny-1, 4] - g[i, 1:Ny-1, 2]) / 2

    return g_plus, g_minus

@njit
def lower_wall_temp(Nx, g_plus, g_minus, g_star):
    j = 0

    g_plus[1:Nx-1, j, 0] = g_star[1:Nx-1, j, 0]
    g_plus[1:Nx-1, j, 1] = (g_star[0:Nx-2, j, 1] + g_star[2:Nx, j, 3]) / 2
    g_plus[1:Nx-1, j, 2] = (g_star[1:Nx-1, j, 4] + g_star[1:Nx-1, j+1, 4]) / 2      # Bounce
    g_plus[1:Nx-1, j, 3] = g_plus[1:Nx-1, j, 1]
    g_plus[1:Nx-1, j, 4] = g_plus[1:Nx-1, j, 2]

    g_minus[1:Nx-1, j, 0] = 0
    g_minus[1:Nx-1, j, 1] = (g_star[0:Nx-2, j, 1] - g_star[2:Nx, j, 3]) / 2
    g_minus[1:Nx-1, j, 2] = (g_star[1:Nx-1, j, 4] - g_star[1:Nx-1, j+1, 4]) / 2      # Bounce
    g_minus[1:Nx-1, j, 3] = -g_minus[1:Nx-1, j, 1]
    g_minus[1:Nx-1, j, 4] = -g_minus[1:Nx-1, j, 2]

    return g_plus, g_minus

@njit
def upper_wall_temp(Nx, Ny, g_plus, g_minus, g_star):
    j = Ny - 1

    g_plus[1:Nx-1, j, 0] = g_star[1:Nx-1, j, 0]
    g_plus[1:Nx-1, j, 1] = (g_star[0:Nx-2, j, 1] + g_star[2:Nx, j, 3]) / 2
    g_plus[1:Nx-1, j, 2] = (g_star[1:Nx-1, j-1, 2] + g_star[1:Nx-1, j, 2]) / 2
    g_plus[1:Nx-1, j, 3] = g_plus[1:Nx-1, j, 1]
    g_plus[1:Nx-1, j, 4] = g_plus[1:Nx-1, j, 2]

    g_minus[1:Nx-1, j, 0] = 0
    g_minus[1:Nx-1, j, 1] = (g_star[0:Nx-2, j, 1] - g_star[2:Nx, j, 3]) / 2
    g_minus[1:Nx-1, j, 2] = (g_star[1:Nx-1, j-1, 2] - g_star[1:Nx-1, j, 2]) / 2
    g_minus[1:Nx-1, j, 3] = -g_minus[1:Nx-1, j, 1]
    g_minus[1:Nx-1, j, 4] = -g_minus[1:Nx-1, j, 2]

    return g_plus, g_minus

@njit
def lower_left_corner_temp(g_plus, g_minus, g_star, w, T):
    i = 0
    j = 0

    g_plus[i, j, 0] = g_star[i, j, 0]
    g_plus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] + g_star[i+1, j, 3]) / 2
    g_plus[i, j, 2] = (g_star[i, j, 4] + g_star[i, j+1, 4]) / 2
    g_plus[i, j, 3] = g_plus[i, j, 1]
    g_plus[i, j, 4] = g_plus[i, j, 2]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] - g_star[i+1, j, 3]) / 2
    g_minus[i, j, 2] = (g_star[i, j, 4] - g_star[i, j+1, 4]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]

    return g_plus, g_minus

@njit
def lower_right_corner_temp(Nx, g_plus, g_minus, g_star, w, T):
    i = Nx - 1
    j = 0

    g_plus[i, j, 0] = g_star[i, j, 0]
    g_plus[i, j, 1] = (g_star[i-1, j, 1] - g_star[i, j, 1] + 2 * w[1] * T[j]) / 2
    g_plus[i, j, 2] = (g_star[i, j, 4] + g_star[i, j+1, 4]) / 2
    g_plus[i, j, 3] = g_plus[i, j, 1]
    g_plus[i, j, 4] = g_plus[i, j, 2]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (g_star[i-1, j, 1] + g_star[i, j, 1] - 2 * w[1] * T[j]) / 2
    g_minus[i, j, 2] = (g_star[i, j, 4] - g_star[i, j+1, 4]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]

    return g_plus, g_minus

@njit
def upper_left_corner_temp(Ny, g_plus, g_minus, g_star, w, T):
    i = 0
    j = Ny - 1

    g_plus[i, j, 0] = g_star[i, j, 0]
    g_plus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] + g_star[i+1, j, 3]) / 2
    g_plus[i, j, 2] = (g_star[i, j-1, 2] + g_star[i, j, 2]) / 2
    g_plus[i, j, 3] = g_plus[i, j, 1]
    g_plus[i, j, 4] = g_plus[i, j, 2]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] - g_star[i+1, j, 3]) / 2
    g_minus[i, j, 2] = (g_star[i, j-1, 2] - g_star[i, j, 2]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]

    return g_plus, g_minus

@njit
def upper_right_corner_temp(Nx, Ny, g_plus, g_minus, g_star, w, T):
    i = Nx - 1
    j = Ny - 1

    g_plus[i, j, 0] = g_star[i, j, 0]
    g_plus[i, j, 1] = (g_star[i-1, j, 1] - g_star[i, j, 1] + 2 * w[1] * T[j]) / 2
    g_plus[i, j, 2] = (g_star[i, j-1, 2] + g_star[i, j, 2]) / 2
    g_plus[i, j, 3] = g_plus[i, j, 1]
    g_plus[i, j, 4] = g_plus[i, j, 2]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (g_star[i-1, j, 1] + g_star[i, j, 1] - 2 * w[1] * T[j]) / 2
    g_minus[i, j, 2] = (g_star[i, j-1, 2] - g_star[i, j, 2]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]

    return g_plus, g_minus
