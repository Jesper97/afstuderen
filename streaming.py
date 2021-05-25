from numba import njit

@njit
def fluid(Nx, Ny, f_plus, f_minus, f_star):
    for i in range(1, Nx-1):

        f = f_plus.copy()

        f[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
        f[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
        f[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
        f[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
        f[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
        f[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 5]
        f[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
        f[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
        f[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

        # f_plus[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
        # f_plus[i, 1:Ny-1, 1] = (f_star[i-1, 1:Ny-1, 1] + f_star[i+1, 1:Ny-1, 3]) / 2
        # f_plus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] + f_star[i, 2:Ny, 4]) / 2
        # f_plus[i, 1:Ny-1, 3] = f_plus[i, 1:Ny-1, 1]
        # f_plus[i, 1:Ny-1, 4] = f_plus[i, 1:Ny-1, 2]
        # f_plus[i, 1:Ny-1, 5] = (f_star[i-1, 0:Ny-2, 5] + f_star[i+1, 2:Ny, 7]) / 2
        # f_plus[i, 1:Ny-1, 6] = (f_star[i+1, 0:Ny-2, 6] + f_star[i-1, 2:Ny, 8]) / 2
        # f_plus[i, 1:Ny-1, 7] = f_plus[i, 1:Ny-1, 5]
        # f_plus[i, 1:Ny-1, 8] = f_plus[i, 1:Ny-1, 6]
        #
        # f_minus[i, 1:Ny-1, 0] = 0
        # f_minus[i, 1:Ny-1, 1] = (f_star[i-1, 1:Ny-1, 1] - f_star[i+1, 1:Ny-1, 3]) / 2
        # f_minus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] - f_star[i, 2:Ny, 4]) / 2
        # f_minus[i, 1:Ny-1, 3] = -f_minus[i, 1:Ny-1, 1]
        # f_minus[i, 1:Ny-1, 4] = -f_minus[i, 1:Ny-1, 2]
        # f_minus[i, 1:Ny-1, 5] = (f_star[i-1, 0:Ny-2, 5] - f_star[i+1, 2:Ny, 7]) / 2
        # f_minus[i, 1:Ny-1, 6] = (f_star[i+1, 0:Ny-2, 6] - f_star[i-1, 2:Ny, 8]) / 2
        # f_minus[i, 1:Ny-1, 7] = -f_minus[i, 1:Ny-1, 5]
        # f_minus[i, 1:Ny-1, 8] = -f_minus[i, 1:Ny-1, 6]

        f_plus[i, 1:Ny-1, 0] = f[i, 1:Ny-1, 0]
        f_plus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] + f[i, 1:Ny-1, 3]) / 2
        f_plus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] + f[i, 1:Ny-1, 4]) / 2
        f_plus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] + f[i, 1:Ny-1, 1]) / 2
        f_plus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] + f[i, 1:Ny-1, 2]) / 2
        f_plus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] + f[i, 1:Ny-1, 7]) / 2
        f_plus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] + f[i, 1:Ny-1, 8]) / 2
        f_plus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] + f[i, 1:Ny-1, 5]) / 2
        f_plus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] + f[i, 1:Ny-1, 6]) / 2

        f_minus[i, 1:Ny-1, 0] = 0
        f_minus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] - f[i, 1:Ny-1, 3]) / 2
        f_minus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] - f[i, 1:Ny-1, 4]) / 2
        f_minus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] - f[i, 1:Ny-1, 1]) / 2
        f_minus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] - f[i, 1:Ny-1, 2]) / 2
        f_minus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] - f[i, 1:Ny-1, 7]) / 2
        f_minus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] - f[i, 1:Ny-1, 8]) / 2
        f_minus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] - f[i, 1:Ny-1, 5]) / 2
        f_minus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] - f[i, 1:Ny-1, 6]) / 2

    return f_plus, f_minus

@njit
def left_wall(Ny, f_plus, f_minus, f_star):
    i = 0

    f = f_plus.copy()

    f[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f[i, 1:Ny-1, 1] = f_star[i, 1:Ny-1, 3]
    f[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f[i, 1:Ny-1, 3] = f_star[i+1, 1:Ny-1, 3]
    f[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f[i, 1:Ny-1, 5] = f_star[i, 1:Ny-1, 7]
    f[i, 1:Ny-1, 6] = f_star[i+1, 0:Ny-2, 6]
    f[i, 1:Ny-1, 7] = f_star[i+1, 2:Ny, 7]
    f[i, 1:Ny-1, 8] = f_star[i, 1:Ny-1, 6]

    # f_plus[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    # f_plus[i, 1:Ny-1, 1] = (f_star[i, 1:Ny-1, 3] + f_star[i+1, 1:Ny-1, 3]) / 2      # Bounce
    # f_plus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] + f_star[i, 2:Ny, 4]) / 2
    # f_plus[i, 1:Ny-1, 3] = f_plus[i, 1:Ny-1, 1]
    # f_plus[i, 1:Ny-1, 4] = f_plus[i, 1:Ny-1, 2]
    # f_plus[i, 1:Ny-1, 5] = (f_star[i, 1:Ny-1, 7] + f_star[i+1, 2:Ny, 7]) / 2        # Bounce
    # f_plus[i, 1:Ny-1, 6] = (f_star[i+1, 0:Ny-2, 6] + f_star[i, 1:Ny-1, 6]) / 2
    # f_plus[i, 1:Ny-1, 7] = f_plus[i, 1:Ny-1, 5]
    # f_plus[i, 1:Ny-1, 8] = f_plus[i, 1:Ny-1, 6]                                     # Bounce
    #
    # f_minus[i, 1:Ny-1, 0] = 0
    # f_minus[i, 1:Ny-1, 1] = (f_star[i, 1:Ny-1, 3] - f_star[i+1, 1:Ny-1, 3]) / 2     # Bounce
    # f_minus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] - f_star[i, 2:Ny, 4]) / 2
    # f_minus[i, 1:Ny-1, 3] = -f_minus[i, 1:Ny-1, 1]
    # f_minus[i, 1:Ny-1, 4] = -f_minus[i, 1:Ny-1, 2]
    # f_minus[i, 1:Ny-1, 5] = (f_star[i, 1:Ny-1, 7] - f_star[i+1, 2:Ny, 7]) / 2       # Bounce
    # f_minus[i, 1:Ny-1, 6] = (f_star[i+1, 0:Ny-2, 6] - f_star[i, 1:Ny-1, 6]) / 2
    # f_minus[i, 1:Ny-1, 7] = -f_minus[i, 1:Ny-1, 5]
    # f_minus[i, 1:Ny-1, 8] = -f_minus[i, 1:Ny-1, 6]                                  # Bounce

    f_plus[i, 1:Ny-1, 0] = f[i, 1:Ny-1, 0]
    f_plus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] + f[i, 1:Ny-1, 3]) / 2
    f_plus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] + f[i, 1:Ny-1, 4]) / 2
    f_plus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] + f[i, 1:Ny-1, 1]) / 2
    f_plus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] + f[i, 1:Ny-1, 2]) / 2
    f_plus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] + f[i, 1:Ny-1, 7]) / 2
    f_plus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] + f[i, 1:Ny-1, 8]) / 2
    f_plus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] + f[i, 1:Ny-1, 5]) / 2
    f_plus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] + f[i, 1:Ny-1, 6]) / 2

    f_minus[i, 1:Ny-1, 0] = 0
    f_minus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] - f[i, 1:Ny-1, 3]) / 2
    f_minus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] - f[i, 1:Ny-1, 4]) / 2
    f_minus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] - f[i, 1:Ny-1, 1]) / 2
    f_minus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] - f[i, 1:Ny-1, 2]) / 2
    f_minus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] - f[i, 1:Ny-1, 7]) / 2
    f_minus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] - f[i, 1:Ny-1, 8]) / 2
    f_minus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] - f[i, 1:Ny-1, 5]) / 2
    f_minus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] - f[i, 1:Ny-1, 6]) / 2

    return f_plus, f_minus

@njit
def right_wall(Nx, Ny, f_plus, f_minus, f_star):
    i = Nx - 1

    f = f_plus.copy()

    f[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    f[i, 1:Ny-1, 1] = f_star[i-1, 1:Ny-1, 1]
    f[i, 1:Ny-1, 2] = f_star[i, 0:Ny-2, 2]
    f[i, 1:Ny-1, 3] = f_star[i, 1:Ny-1, 1]
    f[i, 1:Ny-1, 4] = f_star[i, 2:Ny, 4]
    f[i, 1:Ny-1, 5] = f_star[i-1, 0:Ny-2, 7]
    f[i, 1:Ny-1, 6] = f_star[i, 1:Ny-1, 8]
    f[i, 1:Ny-1, 7] = f_star[i, 1:Ny-1, 5]
    f[i, 1:Ny-1, 8] = f_star[i-1, 2:Ny, 8]

    # f_plus[i, 1:Ny-1, 0] = f_star[i, 1:Ny-1, 0]
    # f_plus[i, 1:Ny-1, 1] = (f_star[i-1, 1:Ny-1, 1] + f_star[i, 1:Ny-1, 1]) / 2      # Bounce
    # f_plus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] + f_star[i, 2:Ny, 4]) / 2
    # f_plus[i, 1:Ny-1, 3] = f_plus[i, 1:Ny-1, 1]
    # f_plus[i, 1:Ny-1, 4] = f_plus[i, 1:Ny-1, 2]
    # f_plus[i, 1:Ny-1, 5] = (f_star[i-1, 0:Ny-2, 5] + f_star[i, 1:Ny-1, 5]) / 2
    # f_plus[i, 1:Ny-1, 6] = (f_star[i, 1:Ny-1, 8] + f_star[i-1, 2:Ny, 8]) / 2        # Bounce
    # f_plus[i, 1:Ny-1, 7] = f_plus[i, 1:Ny-1, 5]                                     # Bounce
    # f_plus[i, 1:Ny-1, 8] = f_plus[i, 1:Ny-1, 6]
    #
    # f_minus[i, 1:Ny-1, 0] = 0
    # f_minus[i, 1:Ny-1, 1] = (f_star[i-1, 1:Ny-1, 1] - f_star[i, 1:Ny-1, 1]) / 2     # Bounce
    # f_minus[i, 1:Ny-1, 2] = (f_star[i, 0:Ny-2, 2] - f_star[i, 2:Ny, 4]) / 2
    # f_minus[i, 1:Ny-1, 3] = -f_minus[i, 1:Ny-1, 1]
    # f_minus[i, 1:Ny-1, 4] = -f_minus[i, 1:Ny-1, 2]
    # f_minus[i, 1:Ny-1, 5] = (f_star[i-1, 0:Ny-2, 5] - f_star[i, 1:Ny-1, 5]) / 2
    # f_minus[i, 1:Ny-1, 6] = (f_star[i, 1:Ny-1, 8] - f_star[i-1, 2:Ny, 8]) / 2       # Bounce
    # f_minus[i, 1:Ny-1, 7] = -f_minus[i, 1:Ny-1, 5]                                  # Bounce
    # f_minus[i, 1:Ny-1, 8] = -f_minus[i, 1:Ny-1, 6]

    f_plus[i, 1:Ny-1, 0] = f[i, 1:Ny-1, 0]
    f_plus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] + f[i, 1:Ny-1, 3]) / 2
    f_plus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] + f[i, 1:Ny-1, 4]) / 2
    f_plus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] + f[i, 1:Ny-1, 1]) / 2
    f_plus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] + f[i, 1:Ny-1, 2]) / 2
    f_plus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] + f[i, 1:Ny-1, 7]) / 2
    f_plus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] + f[i, 1:Ny-1, 8]) / 2
    f_plus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] + f[i, 1:Ny-1, 5]) / 2
    f_plus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] + f[i, 1:Ny-1, 6]) / 2

    f_minus[i, 1:Ny-1, 0] = 0
    f_minus[i, 1:Ny-1, 1] = (f[i, 1:Ny-1, 1] - f[i, 1:Ny-1, 3]) / 2
    f_minus[i, 1:Ny-1, 2] = (f[i, 1:Ny-1, 2] - f[i, 1:Ny-1, 4]) / 2
    f_minus[i, 1:Ny-1, 3] = (f[i, 1:Ny-1, 3] - f[i, 1:Ny-1, 1]) / 2
    f_minus[i, 1:Ny-1, 4] = (f[i, 1:Ny-1, 4] - f[i, 1:Ny-1, 2]) / 2
    f_minus[i, 1:Ny-1, 5] = (f[i, 1:Ny-1, 5] - f[i, 1:Ny-1, 7]) / 2
    f_minus[i, 1:Ny-1, 6] = (f[i, 1:Ny-1, 6] - f[i, 1:Ny-1, 8]) / 2
    f_minus[i, 1:Ny-1, 7] = (f[i, 1:Ny-1, 7] - f[i, 1:Ny-1, 5]) / 2
    f_minus[i, 1:Ny-1, 8] = (f[i, 1:Ny-1, 8] - f[i, 1:Ny-1, 6]) / 2

    return f_plus, f_minus

@njit
def lower_wall(Nx, f_plus, f_minus, f_star):
    j = 0

    f = f_plus.copy()

    f[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f[1:Nx-1, j, 2] = f_star[1:Nx-1, j, 4]
    f[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f[1:Nx-1, j, 4] = f_star[1:Nx-1, j+1, 4]
    f[1:Nx-1, j, 5] = f_star[1:Nx-1, j, 7]
    f[1:Nx-1, j, 6] = f_star[1:Nx-1, j, 8]
    f[1:Nx-1, j, 7] = f_star[2:Nx, j+1, 7]
    f[1:Nx-1, j, 8] = f_star[0:Nx-2, j+1, 8]

    # f_plus[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    # f_plus[1:Nx-1, j, 1] = (f_star[0:Nx-2, j, 1] + f_star[2:Nx, j, 3]) / 2
    # f_plus[1:Nx-1, j, 2] = (f_star[1:Nx-1, j, 4] + f_star[1:Nx-1, j+1, 4]) / 2      # Bounce
    # f_plus[1:Nx-1, j, 3] = f_plus[1:Nx-1, j, 1]
    # f_plus[1:Nx-1, j, 4] = f_plus[1:Nx-1, j, 2]
    # f_plus[1:Nx-1, j, 5] = (f_star[1:Nx-1, j, 7] + f_star[2:Nx, j+1, 7]) / 2        # Bounce
    # f_plus[1:Nx-1, j, 6] = (f_star[1:Nx-1, j, 8] + f_star[0:Nx-2, j+1, 8]) / 2      # Bounce
    # f_plus[1:Nx-1, j, 7] = f_plus[1:Nx-1, j, 5]
    # f_plus[1:Nx-1, j, 8] = f_plus[1:Nx-1, j, 6]
    # 
    # f_minus[1:Nx-1, j, 0] = 0
    # f_minus[1:Nx-1, j, 1] = (f_star[0:Nx-2, j, 1] - f_star[2:Nx, j, 3]) / 2
    # f_minus[1:Nx-1, j, 2] = (f_star[1:Nx-1, j, 4] - f_star[1:Nx-1, j+1, 4]) / 2      # Bounce
    # f_minus[1:Nx-1, j, 3] = -f_minus[1:Nx-1, j, 1]
    # f_minus[1:Nx-1, j, 4] = -f_minus[1:Nx-1, j, 2]
    # f_minus[1:Nx-1, j, 5] = (f_star[1:Nx-1, j, 7] - f_star[2:Nx, j+1, 7]) / 2        # Bounce
    # f_minus[1:Nx-1, j, 6] = (f_star[1:Nx-1, j, 8] - f_star[0:Nx-2, j+1, 8]) / 2      # Bounce
    # f_minus[1:Nx-1, j, 7] = -f_minus[1:Nx-1, j, 5]
    # f_minus[1:Nx-1, j, 8] = -f_minus[1:Nx-1, j, 6]
    
    f_plus[1:Nx-1, j, 0] = f[1:Nx-1, j, 0]
    f_plus[1:Nx-1, j, 1] = (f[1:Nx-1, j, 1] + f[1:Nx-1, j, 1]) / 2
    f_plus[1:Nx-1, j, 2] = (f[1:Nx-1, j, 2] + f[1:Nx-1, j, 2]) / 2
    f_plus[1:Nx-1, j, 3] = (f[1:Nx-1, j, 3] + f[1:Nx-1, j, 3]) / 2
    f_plus[1:Nx-1, j, 4] = (f[1:Nx-1, j, 4] + f[1:Nx-1, j, 4]) / 2
    f_plus[1:Nx-1, j, 5] = (f[1:Nx-1, j, 5] + f[1:Nx-1, j, 5]) / 2
    f_plus[1:Nx-1, j, 6] = (f[1:Nx-1, j, 6] + f[1:Nx-1, j, 6]) / 2
    f_plus[1:Nx-1, j, 7] = (f[1:Nx-1, j, 7] + f[1:Nx-1, j, 7]) / 2
    f_plus[1:Nx-1, j, 8] = (f[1:Nx-1, j, 8] + f[1:Nx-1, j, 8]) / 2

    f_minus[1:Nx-1, j, 0] = 0
    f_minus[1:Nx-1, j, 1] = (f[1:Nx-1, j, 1] - f[1:Nx-1, j, 1]) / 2
    f_minus[1:Nx-1, j, 2] = (f[1:Nx-1, j, 2] - f[1:Nx-1, j, 2]) / 2
    f_minus[1:Nx-1, j, 3] = (f[1:Nx-1, j, 3] - f[1:Nx-1, j, 3]) / 2
    f_minus[1:Nx-1, j, 4] = (f[1:Nx-1, j, 4] - f[1:Nx-1, j, 4]) / 2
    f_minus[1:Nx-1, j, 5] = (f[1:Nx-1, j, 5] - f[1:Nx-1, j, 5]) / 2
    f_minus[1:Nx-1, j, 6] = (f[1:Nx-1, j, 6] - f[1:Nx-1, j, 6]) / 2
    f_minus[1:Nx-1, j, 7] = (f[1:Nx-1, j, 7] - f[1:Nx-1, j, 7]) / 2
    f_minus[1:Nx-1, j, 8] = (f[1:Nx-1, j, 8] - f[1:Nx-1, j, 8]) / 2

    return f_plus, f_minus

@njit
def upper_wall(Nx, Ny, f_plus, f_minus, f_star):
    j = Ny - 1
    
    f = f_plus.copy()

    f[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    f[1:Nx-1, j, 1] = f_star[0:Nx-2, j, 1]
    f[1:Nx-1, j, 2] = f_star[1:Nx-1, j-1, 2]
    f[1:Nx-1, j, 3] = f_star[2:Nx, j, 3]
    f[1:Nx-1, j, 4] = f_star[1:Nx-1, j, 2]
    f[1:Nx-1, j, 5] = f_star[0:Nx-2, j-1, 5]
    f[1:Nx-1, j, 6] = f_star[2:Nx, j-1, 6]
    f[1:Nx-1, j, 7] = f_star[1:Nx-1, j, 5]
    f[1:Nx-1, j, 8] = f_star[1:Nx-1, j, 6]

    # f_plus[1:Nx-1, j, 0] = f_star[1:Nx-1, j, 0]
    # f_plus[1:Nx-1, j, 1] = (f_star[0:Nx-2, j, 1] + f_star[2:Nx, j, 3]) / 2
    # f_plus[1:Nx-1, j, 2] = (f_star[1:Nx-1, j-1, 2] + f_star[1:Nx-1, j, 2]) / 2
    # f_plus[1:Nx-1, j, 3] = f_plus[1:Nx-1, j, 1]
    # f_plus[1:Nx-1, j, 4] = f_plus[1:Nx-1, j, 2]
    # f_plus[1:Nx-1, j, 5] = (f_star[0:Nx-2, j-1, 5] + f_star[1:Nx-1, j, 5]) / 2
    # f_plus[1:Nx-1, j, 6] = (f_star[2:Nx, j-1, 6] + f_star[1:Nx-1, j, 6]) / 2
    # f_plus[1:Nx-1, j, 7] = f_plus[1:Nx-1, j, 5]
    # f_plus[1:Nx-1, j, 8] = f_plus[1:Nx-1, j, 6]
    # 
    # f_minus[1:Nx-1, j, 0] = 0
    # f_minus[1:Nx-1, j, 1] = (f_star[0:Nx-2, j, 1] - f_star[2:Nx, j, 3]) / 2
    # f_minus[1:Nx-1, j, 2] = (f_star[1:Nx-1, j-1, 2] - f_star[1:Nx-1, j, 2]) / 2
    # f_minus[1:Nx-1, j, 3] = -f_minus[1:Nx-1, j, 1]
    # f_minus[1:Nx-1, j, 4] = -f_minus[1:Nx-1, j, 2]
    # f_minus[1:Nx-1, j, 5] = (f_star[0:Nx-2, j-1, 5] - f_star[1:Nx-1, j, 5]) / 2
    # f_minus[1:Nx-1, j, 6] = (f_star[2:Nx, j-1, 6] - f_star[1:Nx-1, j, 6]) / 2
    # f_minus[1:Nx-1, j, 7] = -f_minus[1:Nx-1, j, 5]
    # f_minus[1:Nx-1, j, 8] = -f_minus[1:Nx-1, j, 6]
    
    f_plus[1:Nx-1, j, 0] = f[1:Nx-1, j, 0]
    f_plus[1:Nx-1, j, 1] = (f[1:Nx-1, j, 1] + f[1:Nx-1, j, 1]) / 2
    f_plus[1:Nx-1, j, 2] = (f[1:Nx-1, j, 2] + f[1:Nx-1, j, 2]) / 2
    f_plus[1:Nx-1, j, 3] = (f[1:Nx-1, j, 3] + f[1:Nx-1, j, 3]) / 2
    f_plus[1:Nx-1, j, 4] = (f[1:Nx-1, j, 4] + f[1:Nx-1, j, 4]) / 2
    f_plus[1:Nx-1, j, 5] = (f[1:Nx-1, j, 5] + f[1:Nx-1, j, 5]) / 2
    f_plus[1:Nx-1, j, 6] = (f[1:Nx-1, j, 6] + f[1:Nx-1, j, 6]) / 2
    f_plus[1:Nx-1, j, 7] = (f[1:Nx-1, j, 7] + f[1:Nx-1, j, 7]) / 2
    f_plus[1:Nx-1, j, 8] = (f[1:Nx-1, j, 8] + f[1:Nx-1, j, 8]) / 2

    f_minus[1:Nx-1, j, 0] = 0
    f_minus[1:Nx-1, j, 1] = (f[1:Nx-1, j, 1] - f[1:Nx-1, j, 1]) / 2
    f_minus[1:Nx-1, j, 2] = (f[1:Nx-1, j, 2] - f[1:Nx-1, j, 2]) / 2
    f_minus[1:Nx-1, j, 3] = (f[1:Nx-1, j, 3] - f[1:Nx-1, j, 3]) / 2
    f_minus[1:Nx-1, j, 4] = (f[1:Nx-1, j, 4] - f[1:Nx-1, j, 4]) / 2
    f_minus[1:Nx-1, j, 5] = (f[1:Nx-1, j, 5] - f[1:Nx-1, j, 5]) / 2
    f_minus[1:Nx-1, j, 6] = (f[1:Nx-1, j, 6] - f[1:Nx-1, j, 6]) / 2
    f_minus[1:Nx-1, j, 7] = (f[1:Nx-1, j, 7] - f[1:Nx-1, j, 7]) / 2
    f_minus[1:Nx-1, j, 8] = (f[1:Nx-1, j, 8] - f[1:Nx-1, j, 8]) / 2

    return f_plus, f_minus

@njit
def lower_left_corner(f_plus, f_minus, f_star):
    i = 0
    j = 0
    
    f = f_plus.copy()
    
    f[i, j, 0] = f_star[i, j, 0]
    f[i, j, 1] = f_star[i, j, 3]
    f[i, j, 2] = f_star[i, j, 4]
    f[i, j, 3] = f_star[i+1, j, 3]
    f[i, j, 4] = f_star[i, j+1, 4]
    f[i, j, 5] = f_star[i, j, 7]
    f[i, j, 6] = f_star[i, j, 8]
    f[i, j, 7] = f_star[i+1, j+1, 7]
    f[i, j, 8] = f_star[i, j, 6]

    # f_plus[i, j, 0] = f_star[i, j, 0]
    # f_plus[i, j, 1] = (f_star[i, j, 3] + f_star[i+1, j, 3]) / 2
    # f_plus[i, j, 2] = (f_star[i, j, 4] + f_star[i, j+1, 4]) / 2
    # f_plus[i, j, 3] = f_plus[i, j, 1]
    # f_plus[i, j, 4] = f_plus[i, j, 2]
    # f_plus[i, j, 5] = (f_star[i, j, 7] + f_star[i+1, j+1, 7]) / 2
    # f_plus[i, j, 6] = (f_star[i, j, 8] + f_star[i, j, 6]) / 2
    # f_plus[i, j, 7] = f_plus[i, j, 5]
    # f_plus[i, j, 8] = f_plus[i, j, 6]
    # 
    # f_minus[i, j, 0] = 0
    # f_minus[i, j, 1] = (f_star[i, j, 3] - f_star[i+1, j, 3]) / 2
    # f_minus[i, j, 2] = (f_star[i, j, 4] - f_star[i, j+1, 4]) / 2
    # f_minus[i, j, 3] = -f_minus[i, j, 1]
    # f_minus[i, j, 4] = -f_minus[i, j, 2]
    # f_minus[i, j, 5] = (f_star[i, j, 7] - f_star[i+1, j+1, 7]) / 2
    # f_minus[i, j, 6] = (f_star[i, j, 8] - f_star[i, j, 6]) / 2
    # f_minus[i, j, 7] = -f_minus[i, j, 5]
    # f_minus[i, j, 8] = -f_minus[i, j, 6]
    
    f_plus[i, j, 0] = f[i, j, 0]
    f_plus[i, j, 1] = (f[i, j, 1] + f[i, j, 1]) / 2
    f_plus[i, j, 2] = (f[i, j, 2] + f[i, j, 2]) / 2
    f_plus[i, j, 3] = (f[i, j, 3] + f[i, j, 3]) / 2
    f_plus[i, j, 4] = (f[i, j, 4] + f[i, j, 4]) / 2
    f_plus[i, j, 5] = (f[i, j, 5] + f[i, j, 5]) / 2
    f_plus[i, j, 6] = (f[i, j, 6] + f[i, j, 6]) / 2
    f_plus[i, j, 7] = (f[i, j, 7] + f[i, j, 7]) / 2
    f_plus[i, j, 8] = (f[i, j, 8] + f[i, j, 8]) / 2

    f_minus[i, j, 0] = 0
    f_minus[i, j, 1] = (f[i, j, 1] - f[i, j, 1]) / 2
    f_minus[i, j, 2] = (f[i, j, 2] - f[i, j, 2]) / 2
    f_minus[i, j, 3] = (f[i, j, 3] - f[i, j, 3]) / 2
    f_minus[i, j, 4] = (f[i, j, 4] - f[i, j, 4]) / 2
    f_minus[i, j, 5] = (f[i, j, 5] - f[i, j, 5]) / 2
    f_minus[i, j, 6] = (f[i, j, 6] - f[i, j, 6]) / 2
    f_minus[i, j, 7] = (f[i, j, 7] - f[i, j, 7]) / 2
    f_minus[i, j, 8] = (f[i, j, 8] - f[i, j, 8]) / 2

    return f_plus, f_minus

@njit
def lower_right_corner(Nx, f_plus, f_minus, f_star):
    i = Nx - 1
    j = 0
    
    f = f_plus.copy()
    
    f[i, j, 0] = f_star[i, j, 0]
    f[i, j, 1] = f_star[i-1, j, 1]
    f[i, j, 2] = f_star[i, j, 4]
    f[i, j, 3] = f_star[i, j, 1]
    f[i, j, 4] = f_star[i, j+1, 4]
    f[i, j, 5] = f_star[i, j, 7]
    f[i, j, 6] = f_star[i, j, 8]
    f[i, j, 7] = f_star[i, j, 5]
    f[i, j, 8] = f_star[i-1, j+1, 8]

    # f_plus[i, j, 0] = f_star[i, j, 0]
    # f_plus[i, j, 1] = (f_star[i-1, j, 1] + f_star[i, j, 1]) / 2
    # f_plus[i, j, 2] = (f_star[i, j, 4] + f_star[i, j+1, 4]) / 2
    # f_plus[i, j, 3] = f_plus[i, j, 1]
    # f_plus[i, j, 4] = f_plus[i, j, 2]
    # f_plus[i, j, 5] = (f_star[i, j, 7] + f_star[i, j, 5]) / 2
    # f_plus[i, j, 6] = (f_star[i, j, 8] + f_star[i-1, j+1, 8]) / 2
    # f_plus[i, j, 7] = f_plus[i, j, 5]
    # f_plus[i, j, 8] = f_plus[i, j, 6]
    # 
    # f_minus[i, j, 0] = 0
    # f_minus[i, j, 1] = (f_star[i-1, j, 1] - f_star[i, j, 1]) / 2
    # f_minus[i, j, 2] = (f_star[i, j, 4] - f_star[i, j+1, 4]) / 2
    # f_minus[i, j, 3] = -f_minus[i, j, 1]
    # f_minus[i, j, 4] = -f_minus[i, j, 2]
    # f_minus[i, j, 5] = (f_star[i, j, 7] - f_star[i, j, 5]) / 2
    # f_minus[i, j, 6] = (f_star[i, j, 8] - f_star[i-1, j+1, 8]) / 2
    # f_minus[i, j, 7] = -f_minus[i, j, 5]
    # f_minus[i, j, 8] = -f_minus[i, j, 6]
    
    f_plus[i, j, 0] = f[i, j, 0]
    f_plus[i, j, 1] = (f[i, j, 1] + f[i, j, 1]) / 2
    f_plus[i, j, 2] = (f[i, j, 2] + f[i, j, 2]) / 2
    f_plus[i, j, 3] = (f[i, j, 3] + f[i, j, 3]) / 2
    f_plus[i, j, 4] = (f[i, j, 4] + f[i, j, 4]) / 2
    f_plus[i, j, 5] = (f[i, j, 5] + f[i, j, 5]) / 2
    f_plus[i, j, 6] = (f[i, j, 6] + f[i, j, 6]) / 2
    f_plus[i, j, 7] = (f[i, j, 7] + f[i, j, 7]) / 2
    f_plus[i, j, 8] = (f[i, j, 8] + f[i, j, 8]) / 2

    f_minus[i, j, 0] = 0
    f_minus[i, j, 1] = (f[i, j, 1] - f[i, j, 1]) / 2
    f_minus[i, j, 2] = (f[i, j, 2] - f[i, j, 2]) / 2
    f_minus[i, j, 3] = (f[i, j, 3] - f[i, j, 3]) / 2
    f_minus[i, j, 4] = (f[i, j, 4] - f[i, j, 4]) / 2
    f_minus[i, j, 5] = (f[i, j, 5] - f[i, j, 5]) / 2
    f_minus[i, j, 6] = (f[i, j, 6] - f[i, j, 6]) / 2
    f_minus[i, j, 7] = (f[i, j, 7] - f[i, j, 7]) / 2
    f_minus[i, j, 8] = (f[i, j, 8] - f[i, j, 8]) / 2

    return f_plus, f_minus

@njit
def upper_left_corner(Ny, f_plus, f_minus, f_star):
    i = 0
    j = Ny - 1
    
    f = f_plus.copy()
    
    f[i, j, 0] = f_star[i, j, 0]
    f[i, j, 1] = f_star[i, j, 3]
    f[i, j, 2] = f_star[i, j-1, 2]
    f[i, j, 3] = f_star[i+1, j, 3]
    f[i, j, 4] = f_star[i, j, 2]
    f[i, j, 5] = f_star[i, j, 7]
    f[i, j, 6] = f_star[i+1, j-1, 6]
    f[i, j, 7] = f_star[i, j, 5]
    f[i, j, 8] = f_star[i, j, 6]

    # f_plus[i, j, 0] = f_star[i, j, 0]
    # f_plus[i, j, 1] = (f_star[i, j, 3] + f_star[i+1, j, 3]) / 2
    # f_plus[i, j, 2] = (f_star[i, j-1, 2] + f_star[i, j, 2]) / 2
    # f_plus[i, j, 3] = f_plus[i, j, 1]
    # f_plus[i, j, 4] = f_plus[i, j, 2]
    # f_plus[i, j, 5] = (f_star[i, j, 7] + f_star[i, j, 5]) / 2
    # f_plus[i, j, 6] = (f_star[i+1, j-1, 6] + f_star[i, j, 6]) / 2
    # f_plus[i, j, 7] = f_plus[i, j, 5]
    # f_plus[i, j, 8] = f_plus[i, j, 6]
    # 
    # f_minus[i, j, 0] = 0
    # f_minus[i, j, 1] = (f_star[i, j, 3] - f_star[i+1, j, 3]) / 2
    # f_minus[i, j, 2] = (f_star[i, j-1, 2] - f_star[i, j, 2]) / 2
    # f_minus[i, j, 3] = -f_minus[i, j, 1]
    # f_minus[i, j, 4] = -f_minus[i, j, 2]
    # f_minus[i, j, 5] = (f_star[i, j, 7] - f_star[i, j, 5]) / 2
    # f_minus[i, j, 6] = (f_star[i+1, j-1, 6] - f_star[i, j, 6]) / 2
    # f_minus[i, j, 7] = -f_minus[i, j, 5]
    # f_minus[i, j, 8] = -f_minus[i, j, 6]
    
    f_plus[i, j, 0] = f[i, j, 0]
    f_plus[i, j, 1] = (f[i, j, 1] + f[i, j, 1]) / 2
    f_plus[i, j, 2] = (f[i, j, 2] + f[i, j, 2]) / 2
    f_plus[i, j, 3] = (f[i, j, 3] + f[i, j, 3]) / 2
    f_plus[i, j, 4] = (f[i, j, 4] + f[i, j, 4]) / 2
    f_plus[i, j, 5] = (f[i, j, 5] + f[i, j, 5]) / 2
    f_plus[i, j, 6] = (f[i, j, 6] + f[i, j, 6]) / 2
    f_plus[i, j, 7] = (f[i, j, 7] + f[i, j, 7]) / 2
    f_plus[i, j, 8] = (f[i, j, 8] + f[i, j, 8]) / 2

    f_minus[i, j, 0] = 0
    f_minus[i, j, 1] = (f[i, j, 1] - f[i, j, 1]) / 2
    f_minus[i, j, 2] = (f[i, j, 2] - f[i, j, 2]) / 2
    f_minus[i, j, 3] = (f[i, j, 3] - f[i, j, 3]) / 2
    f_minus[i, j, 4] = (f[i, j, 4] - f[i, j, 4]) / 2
    f_minus[i, j, 5] = (f[i, j, 5] - f[i, j, 5]) / 2
    f_minus[i, j, 6] = (f[i, j, 6] - f[i, j, 6]) / 2
    f_minus[i, j, 7] = (f[i, j, 7] - f[i, j, 7]) / 2
    f_minus[i, j, 8] = (f[i, j, 8] - f[i, j, 8]) / 2

    return f_plus, f_minus

@njit
def upper_right_corner(Nx, Ny, f_plus, f_minus, f_star):
    i = Nx - 1
    j = Ny - 1

    f = f_plus.copy()

    f[i, j, 0] = f_star[i, j, 0]
    f[i, j, 1] = f_star[i-1, j, 1]
    f[i, j, 2] = f_star[i, j-1, 4]
    f[i, j, 3] = f_star[i, j, 1]
    f[i, j, 4] = f_star[i, j, 2]
    f[i, j, 5] = f_star[i-1, j-1, 5]
    f[i, j, 6] = f_star[i, j, 8]
    f[i, j, 7] = f_star[i, j, 5]
    f[i, j, 8] = f_star[i, j, 6]

    # f_plus[i, j, 0] = f_star[i, j, 0]
    # f_plus[i, j, 1] = (f_star[i-1, j, 1] + f_star[i, j, 1]) / 2
    # f_plus[i, j, 2] = (f_star[i, j-1, 2] + f_star[i, j, 2]) / 2
    # f_plus[i, j, 3] = f_plus[i, j, 1]
    # f_plus[i, j, 4] = f_plus[i, j, 2]
    # f_plus[i, j, 5] = (f_star[i-1, j-1, 5] + f_star[i, j, 5]) / 2
    # f_plus[i, j, 6] = (f_star[i, j, 8] + f_star[i, j, 6]) / 2
    # f_plus[i, j, 7] = f_plus[i, j, 5]
    # f_plus[i, j, 8] = f_plus[i, j, 6]
    #
    # f_minus[i, j, 0] = 0
    # f_minus[i, j, 1] = (f_star[i-1, j, 1] - f_star[i, j, 1]) / 2
    # f_minus[i, j, 2] = (f_star[i, j-1, 2] - f_star[i, j, 2]) / 2
    # f_minus[i, j, 3] = -f_minus[i, j, 1]
    # f_minus[i, j, 4] = -f_minus[i, j, 2]
    # f_minus[i, j, 5] = (f_star[i-1, j-1, 5] - f_star[i, j, 5]) / 2
    # f_minus[i, j, 6] = (f_star[i, j, 8] - f_star[i, j, 6]) / 2
    # f_minus[i, j, 7] = -f_minus[i, j, 5]
    # f_minus[i, j, 8] = -f_minus[i, j, 6]

    f_plus[i, j, 0] = f[i, j, 0]
    f_plus[i, j, 1] = (f[i, j, 1] + f[i, j, 1]) / 2
    f_plus[i, j, 2] = (f[i, j, 2] + f[i, j, 2]) / 2
    f_plus[i, j, 3] = (f[i, j, 3] + f[i, j, 3]) / 2
    f_plus[i, j, 4] = (f[i, j, 4] + f[i, j, 4]) / 2
    f_plus[i, j, 5] = (f[i, j, 5] + f[i, j, 5]) / 2
    f_plus[i, j, 6] = (f[i, j, 6] + f[i, j, 6]) / 2
    f_plus[i, j, 7] = (f[i, j, 7] + f[i, j, 7]) / 2
    f_plus[i, j, 8] = (f[i, j, 8] + f[i, j, 8]) / 2

    f_minus[i, j, 0] = 0
    f_minus[i, j, 1] = (f[i, j, 1] - f[i, j, 1]) / 2
    f_minus[i, j, 2] = (f[i, j, 2] - f[i, j, 2]) / 2
    f_minus[i, j, 3] = (f[i, j, 3] - f[i, j, 3]) / 2
    f_minus[i, j, 4] = (f[i, j, 4] - f[i, j, 4]) / 2
    f_minus[i, j, 5] = (f[i, j, 5] - f[i, j, 5]) / 2
    f_minus[i, j, 6] = (f[i, j, 6] - f[i, j, 6]) / 2
    f_minus[i, j, 7] = (f[i, j, 7] - f[i, j, 7]) / 2
    f_minus[i, j, 8] = (f[i, j, 8] - f[i, j, 8]) / 2

    return f_plus, f_minus




@njit
def left_wall_temp(Ny, g_plus, g_minus, g_star, w, T_H):
    i = 0

    g_plus[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
    g_plus[i, 1:Ny-1, 1] = (-g_star[i, 1:Ny-1, 3] + g_star[i+1, 1:Ny-1, 3] + 2 * w[1] * T_H[1:Ny-1]) / 2      # Bounce
    g_plus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] + g_star[i, 2:Ny, 4]) / 2
    g_plus[i, 1:Ny-1, 3] = g_plus[i, 1:Ny-1, 1]
    g_plus[i, 1:Ny-1, 4] = g_plus[i, 1:Ny-1, 2]
    g_plus[i, 1:Ny-1, 5] = (-g_star[i, 1:Ny-1, 7] + g_star[i+1, 2:Ny, 7] + 2 * w[5] * T_H[1:Ny-1]) / 2        # Bounce
    g_plus[i, 1:Ny-1, 6] = (-g_star[i, 1:Ny-1, 6] + g_star[i+1, 0:Ny-2, 6] + 2 * w[6] * T_H[1:Ny-1]) / 2
    g_plus[i, 1:Ny-1, 7] = g_plus[i, 1:Ny-1, 5]
    g_plus[i, 1:Ny-1, 8] = g_plus[i, 1:Ny-1, 6]                                     # Bounce

    g_minus[i, 1:Ny-1, 0] = 0
    g_minus[i, 1:Ny-1, 1] = (-g_star[i, 1:Ny-1, 3] - g_star[i+1, 1:Ny-1, 3] + 2 * w[1] * T_H[1:Ny-1]) / 2     # Bounce
    g_minus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] - g_star[i, 2:Ny, 4]) / 2
    g_minus[i, 1:Ny-1, 3] = -g_minus[i, 1:Ny-1, 1]
    g_minus[i, 1:Ny-1, 4] = -g_minus[i, 1:Ny-1, 2]
    g_minus[i, 1:Ny-1, 5] = (-g_star[i, 1:Ny-1, 7] - g_star[i+1, 2:Ny, 7] + 2 * w[5] * T_H[1:Ny-1]) / 2       # Bounce
    g_minus[i, 1:Ny-1, 6] = (g_star[i, 1:Ny-1, 6] + g_star[i+1, 0:Ny-2, 6] - 2 * w[6] * T_H[1:Ny-1]) / 2
    g_minus[i, 1:Ny-1, 7] = -g_minus[i, 1:Ny-1, 5]
    g_minus[i, 1:Ny-1, 8] = -g_minus[i, 1:Ny-1, 6]                                  # Bounce

    return g_plus, g_minus

@njit
def right_wall_temp(Nx, Ny, g_plus, g_minus, g_star, w, T_C):
    i = Nx - 1

    g_plus[i, 1:Ny-1, 0] = g_star[i, 1:Ny-1, 0]
    g_plus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] - g_star[i, 1:Ny-1, 1] + 2 * w[1] * T_C[1:Ny-1]) / 2      # Bounce
    g_plus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] + g_star[i, 2:Ny, 4]) / 2
    g_plus[i, 1:Ny-1, 3] = g_plus[i, 1:Ny-1, 1]
    g_plus[i, 1:Ny-1, 4] = g_plus[i, 1:Ny-1, 2]
    g_plus[i, 1:Ny-1, 5] = (g_star[i-1, 0:Ny-2, 5] - g_star[i, 1:Ny-1, 5] + 2 * w[5] * T_C[1:Ny-1]) / 2        # Bounce
    g_plus[i, 1:Ny-1, 6] = (-g_star[i, 1:Ny-1, 8] + 2 * w[8] * T_C[1:Ny-1] + g_star[i-1, 2:Ny, 8]) / 2
    g_plus[i, 1:Ny-1, 7] = g_plus[i, 1:Ny-1, 5]
    g_plus[i, 1:Ny-1, 8] = g_plus[i, 1:Ny-1, 6]                                     # Bounce

    g_minus[i, 1:Ny-1, 0] = 0
    g_minus[i, 1:Ny-1, 1] = (g_star[i-1, 1:Ny-1, 1] + g_star[i, 1:Ny-1, 1] - 2 * w[1] * T_C[1:Ny-1]) / 2      # Bounce
    g_minus[i, 1:Ny-1, 2] = (g_star[i, 0:Ny-2, 2] - g_star[i, 2:Ny, 4]) / 2
    g_minus[i, 1:Ny-1, 3] = -g_minus[i, 1:Ny-1, 1]
    g_minus[i, 1:Ny-1, 4] = -g_minus[i, 1:Ny-1, 2]
    g_minus[i, 1:Ny-1, 5] = (g_star[i-1, 0:Ny-2, 5] + g_star[i, 1:Ny-1, 5] - 2 * w[5] * T_C[1:Ny-1]) / 2        # Bounce
    g_minus[i, 1:Ny-1, 6] = (-g_star[i, 1:Ny-1, 8] + 2 * w[8] * T_C[1:Ny-1] - g_star[i-1, 2:Ny, 8]) / 2
    g_minus[i, 1:Ny-1, 7] = g_minus[i, 1:Ny-1, 5]
    g_minus[i, 1:Ny-1, 8] = g_minus[i, 1:Ny-1, 6]                                     # Bounce

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
    # g_plus[i, j, 5] = (w[7] * T[j] + g_star[i+1, j+1, 7]) / 2
    g_plus[i, j, 5] = (-g_star[i, j, 7] + 2 * w[7] * T[j] + g_star[i+1, j+1, 7]) / 2
    g_plus[i, j, 6] = (g_star[i, j, 8] - g_star[i, j, 6] + 2 * w[6] * T[j]) / 2
    g_plus[i, j, 7] = g_plus[i, j, 5]
    g_plus[i, j, 8] = g_plus[i, j, 6]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] - g_star[i+1, j, 3]) / 2
    g_minus[i, j, 2] = (g_star[i, j, 4] - g_star[i, j+1, 4]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]
    # g_minus[i, j, 5] = (w[7] * T[j] - g_star[i+1, j+1, 7]) / 2
    g_minus[i, j, 5] = (-g_star[i, j, 7] + 2 * w[7] * T[j] - g_star[i+1, j+1, 7]) / 2
    g_minus[i, j, 6] = (g_star[i, j, 8] + g_star[i, j, 6] - 2 * w[6] * T[j]) / 2
    g_minus[i, j, 7] = -g_minus[i, j, 5]
    g_minus[i, j, 8] = -g_minus[i, j, 6]

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
    g_plus[i, j, 5] = (g_star[i, j, 7] - g_star[i, j, 5] + 2 * w[5] * T[j]) / 2
    g_plus[i, j, 6] = (-g_star[i, j, 8] + 2 * w[5] * T[j] + g_star[i-1, j+1, 8]) / 2
    g_plus[i, j, 7] = g_plus[i, j, 5]
    g_plus[i, j, 8] = g_plus[i, j, 6]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (g_star[i-1, j, 1] + g_star[i, j, 1] - 2 * w[1] * T[j]) / 2
    g_minus[i, j, 2] = (g_star[i, j, 4] - g_star[i, j+1, 4]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]
    g_minus[i, j, 5] = (g_star[i, j, 7] + g_star[i, j, 5] - 2 * w[5] * T[j]) / 2
    g_minus[i, j, 6] = (-g_star[i, j, 8] + 2 * w[5] * T[j] - g_star[i-1, j+1, 8]) / 2
    g_minus[i, j, 7] = -g_minus[i, j, 5]
    g_minus[i, j, 8] = -g_minus[i, j, 6]

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
    g_plus[i, j, 5] = (-g_star[i, j, 7] + 2 * w[7] * T[j] + g_star[i, j, 5]) / 2
    g_plus[i, j, 6] = (g_star[i+1, j-1, 6] - g_star[i, j, 6] + 2 * w[6] * T[j]) / 2
    g_plus[i, j, 7] = g_plus[i, j, 5]
    g_plus[i, j, 8] = g_plus[i, j, 6]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (-g_star[i, j, 3] + 2 * w[3] * T[j] - g_star[i+1, j, 3]) / 2
    g_minus[i, j, 2] = (g_star[i, j-1, 2] - g_star[i, j, 2]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]
    g_minus[i, j, 5] = (-g_star[i, j, 7] + 2 * w[7] * T[j] - g_star[i, j, 5]) / 2
    g_minus[i, j, 6] = (g_star[i+1, j-1, 6] + g_star[i, j, 6] - 2 * w[6] * T[j]) / 2
    g_minus[i, j, 7] = -g_minus[i, j, 5]
    g_minus[i, j, 8] = -g_minus[i, j, 6]

    return g_plus, g_minus

@njit
def upper_right_corner_temp(Nx, Ny, g_plus, g_minus, g_star, w, T):
    i = Nx - 1
    j = Ny - 1

    g_plus[i, j, 0] = g_star[i, j, 0]
    g_plus[i, j, 1] = (g_star[i-1, j, 1] - g_star[i, j, 1] + 2 * w[3] * T[j]) / 2
    g_plus[i, j, 2] = (g_star[i, j-1, 2] + g_star[i, j, 2]) / 2
    g_plus[i, j, 3] = g_plus[i, j, 1]
    g_plus[i, j, 4] = g_plus[i, j, 2]
    g_plus[i, j, 5] = (g_star[i-1, j-1, 5] - g_star[i, j, 5] + 2 * w[5] * T[j]) / 2
    g_plus[i, j, 6] = (-g_star[i, j, 8] + 2 * w[8] * T[j] + g_star[i, j, 6]) / 2
    g_plus[i, j, 7] = g_plus[i, j, 5]
    g_plus[i, j, 8] = g_plus[i, j, 6]

    g_minus[i, j, 0] = 0
    g_minus[i, j, 1] = (g_star[i-1, j, 1] + g_star[i, j, 1] - 2 * w[3] * T[j]) / 2
    g_minus[i, j, 2] = (g_star[i, j-1, 2] - g_star[i, j, 2]) / 2
    g_minus[i, j, 3] = -g_minus[i, j, 1]
    g_minus[i, j, 4] = -g_minus[i, j, 2]
    g_minus[i, j, 5] = (g_star[i-1, j-1, 5] + g_star[i, j, 5] - 2 * w[5] * T[j]) / 2
    g_minus[i, j, 6] = (-g_star[i, j, 8] + 2 * w[8] * T[j] - g_star[i, j, 6]) / 2
    g_minus[i, j, 7] = -g_minus[i, j, 5]
    g_minus[i, j, 8] = -g_minus[i, j, 6]

    return g_plus, g_minus
