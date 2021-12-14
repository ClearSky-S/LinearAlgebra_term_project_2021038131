import numpy as np

a = (1, 2)
b = a
a = (1,3)

print(a)
print(b)
a = np.array([[1,1],[2,2]])
print(a)
print(a.shape)