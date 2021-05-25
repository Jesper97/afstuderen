from numba import njit

@njit
def fluid(Nx, Ny, e, f_new, f_old):
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            for k in range(9):
                ip = i - e[k, 0]
                jp = j - e[k, 1]
                f_new[i, j, k] = f_old[ip, jp, k]
    return f_new


@njit
def left_right_wall(Nx, Ny, f_new, f_old):
    for j in range(1, Ny - 1):
        # Left wall
        i = 0
        f_new[i, j, 0] = f_old[i, j, 0]
        f_new[i, j, 1] = f_old[i, j, 3]
        f_new[i, j, 2] = f_old[i, j - 1, 2]
        f_new[i, j, 3] = f_old[i + 1, j, 3]
        f_new[i, j, 4] = f_old[i, j + 1, 4]
        f_new[i, j, 5] = f_old[i, j, 7]
        f_new[i, j, 6] = f_old[i+1, j-1, 6]
        f_new[i, j, 7] = f_old[i+1, j+1, 7]
        f_new[i, j, 8] = f_old[i, j, 6]

        # Right wall
        i = Nx - 1
        f_new[i, j, 0] = f_old[i, j, 0]
        f_new[i, j, 1] = f_old[i - 1, j, 1]
        f_new[i, j, 2] = f_old[i, j - 1, 2]
        f_new[i, j, 3] = f_old[i, j, 1]
        f_new[i, j, 4] = f_old[i, j + 1, 4]
        f_new[i, j, 5] = f_old[i-1, j-1, 5]
        f_new[i, j, 6] = f_old[i, j, 8]
        f_new[i, j, 7] = f_old[i, j, 5]
        f_new[i, j, 8] = f_old[i-1, j+1, 8]

    return f_new


@njit
def top_bottom_wall(Nx, Ny, f_new, f_old, w, r, bc_value):
    for i in range(1, Nx-1):
        # Bottom wall
        j = 0
        f_new[i, j, 0] = f_old[i, j, 0]
        f_new[i, j, 1] = f_old[i - 1, j, 1]
        f_new[i, j, 2] = f_old[i, j, 4]
        f_new[i, j, 3] = f_old[i + 1, j, 3]
        f_new[i, j, 4] = f_old[i, j + 1, 4]
        f_new[i, j, 5] = f_old[i, j, 7]
        f_new[i, j, 6] = f_old[i, j, 8]
        f_new[i, j, 7] = f_old[i + 1, j + 1, 7]
        f_new[i, j, 8] = f_old[i - 1, j + 1, 8]

        # Top wall
        j = Ny - 1
        f_new[i, j, 0] = f_old[i, j, 0]
        f_new[i, j, 1] = f_old[i - 1, j, 1]
        f_new[i, j, 2] = f_old[i, j - 1, 2]
        f_new[i, j, 3] = f_old[i + 1, j, 3]
        f_new[i, j, 4] = f_old[i, j, 2]
        f_new[i, j, 5] = f_old[i - 1, j - 1, 5]
        f_new[i, j, 6] = f_old[i + 1, j - 1, 6]
        f_new[i, j, 7] = f_old[i, j, 5] #- 6 * w[5] * r[i, j] * bc_value[1, 0]
        f_new[i, j, 8] = f_old[i, j, 6] #+ 6 * w[6] * r[i, j] * bc_value[1, 0]

    return f_new


@njit
def bottom_corners(Nx, Ny, f_new, f_old):
    # Bottom left corner
    i = 0
    j = 0
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i, j, 3]
    f_new[i, j, 2] = f_old[i, j, 4]
    f_new[i, j, 3] = f_old[i+1, j, 3]
    f_new[i, j, 4] = f_old[i, j+1, 4]
    f_new[i, j, 5] = f_old[i, j, 7]
    f_new[i, j, 6] = f_old[i, j, 8]
    f_new[i, j, 7] = f_old[i+1, j+1, 7]
    f_new[i, j, 8] = f_old[i, j, 6]

    # Bottom right corner
    i = Nx - 1
    j = 0
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i-1, j, 1]
    f_new[i, j, 2] = f_old[i, j, 4]
    f_new[i, j, 3] = f_old[i, j, 1]
    f_new[i, j, 4] = f_old[i, j+1, 4]
    f_new[i, j, 5] = f_old[i, j, 7]
    f_new[i, j, 6] = f_old[i, j, 8]
    f_new[i, j, 7] = f_old[i, j, 5]
    f_new[i, j, 8] = f_old[i-1, j+1, 8]

    return f_new


@njit
def top_corners(Nx, Ny, f_new, f_old, w, r, bc_value):
    # Top right corner
    i = Nx - 1
    j = Ny - 1
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i - 1, j, 1]
    f_new[i, j, 2] = f_old[i, j - 1, 2]
    f_new[i, j, 3] = f_old[i, j, 1]
    f_new[i, j, 4] = f_old[i, j, 2]
    f_new[i, j, 5] = f_old[i - 1, j - 1, 5]
    f_new[i, j, 6] = f_old[i, j, 8]
    f_new[i, j, 7] = f_old[i, j, 5] #- 6 * w[5] * r[i, j] * bc_value[1, 0]
    f_new[i, j, 8] = f_old[i, j, 6] #+ 6 * w[6] * r[i, j] * bc_value[1, 0]

    # Top left corner
    i = 0
    j = Ny - 1
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i, j, 3]
    f_new[i, j, 2] = f_old[i, j - 1, 2]
    f_new[i, j, 3] = f_old[i + 1, j, 3]
    f_new[i, j, 4] = f_old[i, j, 2]
    f_new[i, j, 5] = f_old[i, j, 7]
    f_new[i, j, 6] = f_old[i + 1, j - 1, 6]
    f_new[i, j, 7] = f_old[i, j, 5] #- 6 * w[5] * r[i, j] * bc_value[1, 0]
    f_new[i, j, 8] = f_old[i, j, 6] #+ 6 * w[6] * r[i, j] * bc_value[1, 0]

    return f_new

