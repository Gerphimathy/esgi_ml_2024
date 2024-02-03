from importlib import reload

from load_dataset import *
from train_model import *

RGB = True
IMAGE_RESOLUTION = 50 * 74
TRAINING_PERCENTAGE = 0.8
if RGB:
    IMAGE_RESOLUTION *= 3

imgs = index_images()
print("Images indexed")
train_model(IMAGE_RESOLUTION, imgs, TRAINING_PERCENTAGE)

