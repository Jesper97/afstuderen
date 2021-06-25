import numpy as np
import pandas as pd
import sys

# a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# np.savetxt("foo.csv", a, delimiter=",")
#
# x = ".png.csv"
# print(".png.csv".replace(".png", ""))

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

def easy_view(nr, arr):
    idx = ["idx" for i in arr[1, :]]
    col = ["col" for j in arr[:, 1]]

    dataset = pd.DataFrame(arr.T, index=idx, columns=col)
    print(nr, dataset)


Nx = 150
Ny = 35
L1 = 0.1
L2 = 0.2
L = L1 + L2

idx = np.int(L2/L * Nx)

x = np.ones((Nx+2, Ny+2), dtype=int)
x[:, :6] = 2
x[:, -6:] = 2
x[:, 0] = 3
x[:, -1] = 3

idx_boundary = 5
for j in range(Ny+1-idx_boundary, Ny+1):
    print(x[1, j])


easy_view(1, x)
