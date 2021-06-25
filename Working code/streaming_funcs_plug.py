from numba import njit, prange


@njit(parallel=True)
def fluid(bound_node_idx, Nx, Ny, e, f_new, f_old):
    for j in prange(bound_node_idx+1, Ny-bound_node_idx-1):
        for i in prange(1, Nx-1):
            for k in prange(9):
                ip = i - e[k, 0]
                jp = j - e[k, 1]
                f_new[i, j, k] = f_old[ip, jp, k]
    return f_new


@njit(parallel=True)
def left_right_wall(bound_node_idx, Nx, Ny, f_new, f_old):
    for j in prange(bound_node_idx+1, Ny-bound_node_idx-1):
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


@njit(parallel=True)
def top_bottom_wall(bound_node_idx, Nx, Ny, f_new, f_old):
    for i in prange(1, Nx-1):
        # Bottom wall
        j = bound_node_idx
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
        j = Ny - bound_node_idx - 1
        f_new[i, j, 0] = f_old[i, j, 0]
        f_new[i, j, 1] = f_old[i - 1, j, 1]
        f_new[i, j, 2] = f_old[i, j - 1, 2]
        f_new[i, j, 3] = f_old[i + 1, j, 3]
        f_new[i, j, 4] = f_old[i, j, 2]
        f_new[i, j, 5] = f_old[i - 1, j - 1, 5]
        f_new[i, j, 6] = f_old[i + 1, j - 1, 6]
        f_new[i, j, 7] = f_old[i, j, 5]
        f_new[i, j, 8] = f_old[i, j, 6]

    return f_new


@njit
def bottom_corners(bound_node_idx, Nx, Ny, f_new, f_old):
    # Bottom left corner
    i = 0
    j = bound_node_idx
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
    j = bound_node_idx
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
def top_corners(bound_node_idx, Nx, Ny, f_new, f_old):
    # Top right corner
    i = Nx - 1
    j = Ny - bound_node_idx - 1
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i - 1, j, 1]
    f_new[i, j, 2] = f_old[i, j - 1, 2]
    f_new[i, j, 3] = f_old[i, j, 1]
    f_new[i, j, 4] = f_old[i, j, 2]
    f_new[i, j, 5] = f_old[i - 1, j - 1, 5]
    f_new[i, j, 6] = f_old[i, j, 8]
    f_new[i, j, 7] = f_old[i, j, 5]
    f_new[i, j, 8] = f_old[i, j, 6]

    # Top left corner
    i = 0
    j = Ny - bound_node_idx - 1
    f_new[i, j, 0] = f_old[i, j, 0]
    f_new[i, j, 1] = f_old[i, j, 3]
    f_new[i, j, 2] = f_old[i, j - 1, 2]
    f_new[i, j, 3] = f_old[i + 1, j, 3]
    f_new[i, j, 4] = f_old[i, j, 2]
    f_new[i, j, 5] = f_old[i, j, 7]
    f_new[i, j, 6] = f_old[i + 1, j - 1, 6]
    f_new[i, j, 7] = f_old[i, j, 5]
    f_new[i, j, 8] = f_old[i, j, 6]

    return f_new

