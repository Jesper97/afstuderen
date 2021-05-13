import numpy as np
import time

# a = np.array(range(10))
# s = slice(2, 4)
# print(a[s])

z = 2.63

start = time.time()
y = 0.5 * z
start2 = time.time()
x = 1/2 * z
stop = time.time()

print(start2-start)
print(stop-start2)
