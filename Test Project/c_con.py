from ctypes import POINTER, pointer, addressof, c_double, c_int, cast
import ctypes
import numpy as np


def to_cdouble_array(lst):
    if isinstance(lst, list):
        return (c_double * len(lst))(*lst)
    elif isinstance(lst, np.ndarray):
        return (c_double * len(lst))(*lst.tolist())
    else:
        raise ValueError('Input should be a list.')


def to_cint_array(lst):
    if isinstance(lst, list):
        return (c_int * len(lst))(*lst)
    elif isinstance(lst, np.ndarray):
        return (c_int * len(lst))(*lst.tolist())
    else:
        raise ValueError('Input should be a list.')

def np_2d_array_to_cdouble_array(np_array):
    np_array = np_array.astype(np.float64)  # Ensure the numpy array is of type float64
    y, x = np_array.shape

    double_ptr = ctypes.POINTER(ctypes.c_double)
    double_ptr_p = ctypes.POINTER(double_ptr)  # The double pointer (double**)

    arr = (double_ptr * y)()  # The equivalent of double*[y]

    for i in range(y):
        arr[i] = (ctypes.c_double * x)()  # The equivalent of double[x]
        for j in range(x):
            arr[i][j] = np_array[i][j]  # Set the value

    return arr