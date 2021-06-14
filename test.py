import numpy as np

a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt("foo.csv", a, delimiter=",")

x = ".png.csv"
print(".png.csv".replace(".png", ""))
