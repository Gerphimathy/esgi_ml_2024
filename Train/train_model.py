import ctypes
import os
import random
from ctypes import cdll
from math import floor

from load_dataset import *

from c_con import *

def train_model(IMAGE_RESOLUTION, imgs, training_percentage):
    LAYERS = [IMAGE_RESOLUTION, IMAGE_RESOLUTION, 5000, 1000, 3]

    library_path = "./MLCore.dll"
    if os.path.exists(library_path):
        libc = cdll.LoadLibrary(library_path)
    else:
        raise Exception(f"The specified library does not exist: {library_path}")

    print("Library loaded")
    print("Creating model with layers: ", LAYERS)

    libc.Init()
    libc.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

    if os.path.exists(b"model.txt"):
        model = libc.deserialize_mlp(b"model.txt")
    else:
        # ACTIVATION FUNCTIONS:
        # SIGMOID (0),
        # TANH (1),
        # RELU (2),
        # LEAKY_RELU (3),
        # SOFTMAX (4),
        model = libc.create_mlp(to_cint_array(LAYERS), len(LAYERS), 1)

    print("Model created")

    # Train the model
    # SAMPLING:
    # RANDOM (0)
    # BATCH_GRADIANT_DESCENT (1)
    # STOCHASTIC_GRADIANT_DESCENT (2)
    # MINI_BATCH_GRADIANT_DESCENT (3)

    SAMPLING = 0
    TRAINING_RATE = 0.00001
    EPOCHS = 100000
    SAVE_RATE = 1000

    initial_epoch = 0

    for i in range(initial_epoch, EPOCHS):
        print(f"Epoch {i}/{EPOCHS}")

        l = random.randint(0, floor(len(imgs) - 1 * training_percentage))

        IMG_DATA, IMG_CLASSES = load_img(l, IMAGE_RESOLUTION)

        if IMG_DATA is None or IMG_CLASSES is None:
            continue

        libc.train_mlp(model, np_2d_array_to_cdouble_array(IMG_DATA), np_2d_array_to_cdouble_array(IMG_CLASSES),
                       1, True, c_double(TRAINING_RATE), 1, SAMPLING)

        if i % SAVE_RATE == 0 and i > 0:
            libc.serialize_mlp(model, b"model.txt")

    # Represent data as 3 bars, one for each class
    # plt.bar([0, 1, 2], [np.sum(IMG_CLASSES[:, 0]), np.sum(IMG_CLASSES[:, 1]), np.sum(IMG_CLASSES[:, 2])])

    libc.Quit()