import numpy as np
from ctypes import cdll, c_int, c_double
import os
from c_con import *
import matplotlib.pyplot as plt


def MLP_Linear_Simple(libc):
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        [1],
        [-1],
        [-1]
    ])

    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')

    test_model = libc.create_mlp(to_cint_array([2, 2, 2, 1]), 4, 1)
    libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 3, True,
                   c_double(0.0001), 100000, 2)

    # Test the model
    x = np.linspace(1, 3, 100)
    y = np.linspace(1, 3, 100)
    xx, yy = np.meshgrid(x, y)
    pred = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            pred[i, j] = libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True)[0]

    plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
    plt.colorbar()

    plt.show()
    plt.clf()


def MLP_Linear_Multiple(libc):
    test_model = libc.create_mlp(to_cint_array([2, 2, 2, 1]), 4, 1)

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])],
        dtype=np.float64)
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0], dtype=np.float64)

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')

    libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 100, True,
                   c_double(0.00001), 100000, 2)

    # Test the model
    x = np.linspace(1, 3, 100)
    y = np.linspace(1, 3, 100)
    xx, yy = np.meshgrid(x, y)
    pred = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            pred[i, j] = libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True)[0]

    plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
    plt.colorbar()

    plt.show()
    plt.clf()


def MLP_XOR(libc):
    test_model = libc.create_mlp(to_cint_array([2, 2, 2, 1]), 4, 1)
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

    libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 4, True,
                   c_double(0.0001), 100000, 2)

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


def MLP_Cross(libc):
    test_model = libc.create_mlp(to_cint_array([2, 3, 2, 1]), 4, 1)

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in X])

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1],
                color='red')

    libc.train_mlp(test_model, np_2d_array_to_cdouble_array(X), np_2d_array_to_cdouble_array(Y), 500, True,
                   c_double(0.00001), 100000, 2)

    # Test the model
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    pred = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            pred[i, j] = libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True)[0]

    plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
    plt.colorbar()

    plt.show()
    plt.clf()


def MLP_3_Classes(libc):
    test_model = libc.create_mlp(to_cint_array([2, 3, 3, 3]), 4, 1)

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')

    X = np_2d_array_to_cdouble_array(X)
    Y = np_2d_array_to_cdouble_array(Y)

    libc.train_mlp(test_model, X, Y, 500, True, c_double(0.00001), 10000, 2)

    # Test the model, 3 classes classification, red blue green
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    pred = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            pred[i, j] = np.argmax(libc.predict_mlp(test_model, to_cdouble_array([xx[i, j], yy[i, j]]), True))[0]

    plt.contourf(xx, yy, pred, cmap='RdBu', alpha=0.5)
    plt.colorbar()

    plt.show()
    plt.clf()
