from ctypes import cdll, c_int, c_double
import os
from c_con import *
import matplotlib.pyplot as plt

library_path = "./MLCore.dll"
if os.path.exists(library_path):
    libc = cdll.LoadLibrary(library_path)
else:
    raise Exception(f"The specified library does not exist: {library_path}")

libc.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

test_model = libc.create_mlp(to_cint_array([2, 2, 2, 2, 1]), 5, 1)
print(test_model)

X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])], dtype=np.float64)
Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0], dtype=np.float64)

plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
plt.scatter(X[50:100,0], X[50:100,1], color='red')


print(libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 100, True, c_double(0.00001), 100000, 2))

pred_ptr = libc.predict_mlp(test_model, to_cdouble_array([0,0]), True)
pred_values = [pred_ptr[i] for i in range(1)]
print(pred_values)

# Test the model
x = np.linspace(1, 3, 100)
y = np.linspace(1, 3, 100)
xx, yy = np.meshgrid(x, y)
pred = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        pred_ptr = libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True)
        pred[i, j] = pred_ptr[0]

plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
plt.colorbar()

plt.show()
plt.clf()

libc.delete_all_mlps()

