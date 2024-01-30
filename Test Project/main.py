from ctypes import cdll, c_int, c_double
import os
from c_con import *
import matplotlib.pyplot as plt
from tests.classification import *
from tests.regression import *

library_path = "./MLCore.dll"
if os.path.exists(library_path):
    libc = cdll.LoadLibrary(library_path)
else:
    raise Exception(f"The specified library does not exist: {library_path}")

libc.Init()
libc.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

# MLP_Linear_Simple(libc)
# MLP_Linear_Multiple(libc)
# MLP_XOR(libc)
# MLP_Cross(libc)
# MLP_3_Classes(libc)
# MLP_Multi_Cross(libc)
# MLP_Linear_Simple_2D(libc)

test_model = libc.deserialize_mlp(b"test_model.txt")
X = np.array([[1, 0],
              [0, 1],
              [0, 0],
              [1, 1]]
             )
Y = np.array([[1],
              [1],
              [-1],
              [-1]]
             )

plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
plt.scatter(X[2:4, 0], X[2:4, 1], color='red')

# Test the model
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x, y)
pred = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        pred[i, j] = libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True)[0]

plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
plt.colorbar()

plt.show()
plt.clf()

libc.Quit()
