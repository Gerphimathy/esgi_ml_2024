import ctypes
import os
import random
from ctypes import cdll
from math import floor

from load_dataset import *

from c_con import *


def train_model(libc, model, IMAGE_RESOLUTION, imgs, training_percentage, epochs, save_rate, training_rate):
    # Train the model
    # SAMPLING:
    # RANDOM (0)
    # BATCH_GRADIANT_DESCENT (1)
    # STOCHASTIC_GRADIANT_DESCENT (2)
    # MINI_BATCH_GRADIANT_DESCENT (3)
    SAMPLING = 0

    for i in range(epochs):
        print(f"Epoch {i}/{epochs}")

        l = random.randint(0, floor(len(imgs) - 1 * training_percentage))

        IMG_DATA, IMG_CLASSES = load_img(l, IMAGE_RESOLUTION)

        if IMG_DATA is None or IMG_CLASSES is None:
            continue

        libc.train_mlp(model, np_2d_array_to_cdouble_array(IMG_DATA), np_2d_array_to_cdouble_array(IMG_CLASSES),
                       1, True, c_double(training_rate), 1, SAMPLING)

        if i % save_rate == 0 and i > 0 and i < epochs - 1:
            libc.serialize_mlp(model, b"model.txt")

    libc.serialize_mlp(model, b"model.txt")
