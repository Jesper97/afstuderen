# Programming example of tensordot when axes is an scalar
import numpy as np
# Declaring arrays
# arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# arr2 = np.arange(1, 5).reshape(2, 2)
# print(arr1.shape)
# print(arr2.shape)
# print("Tensor1 is: \n", arr1)
# print("\nTensor2 is: \n", arr2)
#
# # Now we will calculate tensor dot
# ans = np.tensordot(arr2, arr1, axes=1)
# ans2 = np.einsum('ij,klj->kli', arr2, arr1)
# # print(arr1[0, 1, 0])
# print(ans2)
# print(ans2[1, 1])
# # print("Tensordot of these tensors is:\n", ans)
#
# y = np.linspace(0, 40-1, 40)
# print(y)

Nx = 3
x = np.ones((Nx+2, Nx+2))
# x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(x.shape)
print(x[1:Nx+2, 0])
