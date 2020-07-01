import numpy as np

a = np.arange(100)

a = a.reshape(-1, 5)

print(a)

b = [1, 1, 1, 2, 2]

c = a-b 

sum = np.sum(c, axis = 1)

min_index = np.argmin(np.absolute(sum))

print(c)
print(sum)

print(min_index)

print(a[0] - b)

print(np.random.randint(0, 200))