import os.path

from c_con import *
from PIL import Image
import numpy as np


def load_img(line_number, IMAGE_RESOLUTION):
    with open("dataset.csv", 'r') as file:
        if line_number == 0:
            return None, None

        line = file.readlines()[line_number]
        img_slug = line.split(';')[0]
        img_genres = line.split(';')[1].split(',')

        try:
            IMG = Image.open(f"resized_images/{img_slug}.jpg")
        except FileNotFoundError:
            print(f"Dataset : Image {img_slug} not found. Skipping...")
            return None, None

        IMG_DATA = [np.array(IMG.getdata()).flatten()]
        IMG_CLASSES = [np.array([
                1 if "Drama" in img_genres else 0,
                1 if "Comedy" in img_genres else 0,
                1 if "Action" in img_genres else 0]
        )]

        IMG_DATA = np.array(IMG_DATA, np.float64)
        IMG_DATA = IMG_DATA.reshape((1, IMAGE_RESOLUTION))

        IMG_CLASSES = np.array(IMG_CLASSES, np.float64)
        IMG_CLASSES = IMG_CLASSES.reshape((1, 3))

        return IMG_DATA, IMG_CLASSES


def index_images():
    imgs_lines = []
    l = 0
    with open("dataset.csv", 'r') as file:
        print(f"Indexing images...")
        for line in file:
            l += 1
            if line.startswith("slug"):
                continue
            if not os.path.exists(f"resized_images/{line.split(';')[0]}.jpg"):
                continue
            imgs_lines.append(l)

    return imgs_lines
