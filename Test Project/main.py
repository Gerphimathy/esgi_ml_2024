from ctypes import cdll, c_int, c_double
import os
from c_con import *
import matplotlib.pyplot as plt
from tests.classification import *

library_path = "./MLCore.dll"
if os.path.exists(library_path):
    libc = cdll.LoadLibrary(library_path)
else:
    raise Exception(f"The specified library does not exist: {library_path}")

libc.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

MLP_Linear_Simple(libc)
MLP_Linear_Multiple(libc)
MLP_XOR(libc)
MLP_Cross(libc)
MLP_3_Classes(libc)

libc.delete_all_mlps()

