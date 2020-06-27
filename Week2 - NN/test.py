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


# print(A)

# transposeA = np.einsum('ijk->ikj', A)

# print(transposeA)

# print(A.shape)

# D = np.arange(36).reshape(6, 2, 3)
# print(D)

# batch = 2
# D = D.reshape(2, -1, 2, 3)

# print(D)

E = np.arange(6).reshape(2, 3, 1)
F = np.arange(6).reshape(2, 1, 3)

E1 = np.arange(3).reshape(3, 1)
E2 = np.arange(3).reshape(3, 1) + 3

F1 = np.arange(3).reshape(1, 3)
F2 = np.arange(3).reshape(1, 3) + 3

G = E * F

# print(G)

# print(E2 * F2)
# print(np.matmul(E2, F2))

# print(E1.shape)
# print(E1.transpose().shape)

print(A1 * A1)

'''
MLN algorithm:: 

* Input: (m) x 784 x 1

* FCL:   64 neurons
* FCL_w: (64 x 784)
* FCL_b: (64 x 1)

* output layer: 10 neurons
* output_w:     (10 x 64)
* output_b:     (10 x 1)


Forward: 
* x -> Input                                    m x 784 x 1

* z1 = FCL_w * Input + FCL_b                    m x 64 x 1
* a1 = sigmoid(z1)

* z2 = output_w * output_w + output_b                 m x 1 x 10
* y  = sigmoid(z2)                              

Backward: 

Step 1: Error
* output_error  = (y - label) .* d_sigmoid(z2)                          m x 1 x 10

                   (m x 1 x 10)      (64 x 10)^T    m x 1 x 64
* FCL_error     = (output_error * output_w^T) .* (d_sigmoid(z1))        m x 1 x 64


Step 2: Gradient descent

                 (64 x 10)    constant           m x (1 x 10)^T   m x 1 x 64  
* output_w      = output_w - (alpha / m) * sum( output_error^T  *    z1      )^T     64 x 10

                  (1 x 10)    constant           m x 1 x 10
* output_b      = output_b - (alpha / m) * sum( output_error )

                 (784 x 64)   constant           m x (1 x 64)^T   m x 1 x 784  
* FCL_w         = FCL_w    - (alpha / m) * sum( FCL_error^T     *    Input   )^T    784 x 64

                  (1 x 10)    constant           m x 1 x 10
* FCL_b         = FCL_b    - (alpha / m) * sum( fcl_error )


'''

'''

1. Create batches
x_train.reshape(-1, batches, 28, 28) # maybe?

2. For each batch:

        forward pass
        calculate loss
        back propagation

        cross validate on (test_set)?
        test loss 
        test accuracy

'''