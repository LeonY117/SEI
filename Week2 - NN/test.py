import numpy as np

# a = [1, 2, 3, 4, 5]

# b = [2, 3, 4, 5, 6]
# a = np.array(a)
# b = np.array(b)


# def sigmoid(x):
#     return (1 / (1 + np.exp(-x)))

# def d_sigmoid(x):
#     return sigmoid(x) * (1-sigmoid(x))


# print(d_sigmoid(a))

# print(a * b)


A = np.arange(12).reshape(2, 2, 3)

B = np.arange(12).reshape(2, 3, 2)

# C = np.matmul(A, B)

# print(C)

A1 = np.arange(6).reshape(2, 3)
A2 = np.arange(6).reshape(2, 3) + 6

# B1 = np.arange(6).reshape(3, 2)
# B2 = np.arange(6).reshape(3, 2) + 6

# C1 = np.matmul(A1, B1)
# C2 = np.matmul(A2, B2)

# B = np.arange(6).reshape(3, 2)

# C = np.matmul(A, B)

# print(C)
# C1 = np.matmul(A1, B)
# C2 = np.matmul(A2, B)

# print(C1)
# print(C2)


print(A)

transposeA = np.einsum('ijk->ikj', A)

print(transposeA)

print(A1.transpose())