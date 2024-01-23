import numpy as np
from ctypes import cdll, c_int, c_double
import os
from c_con import *
import matplotlib.pyplot as plt


def MLP_Linear_Simple_2D(libc):
    test_model = libc.create_mlp(to_cint_array([1, 1]), 2, 1)

    X = np.array([
        [1],
        [2]
    ])
    Y = np.array([
        [2],
        [3]
    ])

    libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 2, False, c_double(0.001), 100, 2)

    plt.scatter(X, Y, color="blue")

    # Test the model
    x = np.linspace(0, 3, 100)
    pred = np.zeros((100, 1))
    for i in range(100):
        pred[i] = libc.predict_mlp(test_model, to_cdouble_array([x[i]]), False)[0]

    plt.plot(x, pred, color="red")

    plt.show()
    plt.clf()