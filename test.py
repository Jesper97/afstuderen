import numpy as np

x = np.array([[0, 1, 2], [3, 4, 5]])
y = np.array([[0, 1, 3.5], [3, 1, 5]])
z = np.any((abs(x - y) > 1))
print(z)
