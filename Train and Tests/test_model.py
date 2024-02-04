from ctypes import *
import random
from math import floor
from load_dataset import *
from matplotlib import pyplot as plt
import numpy as np


def test_model(libc, model, IMAGE_RESOLUTION, imgs, movie_slug):
    if not os.path.exists(f"resized_images/{movie_slug}.jpg"):
        print(f"Image {movie_slug} not found")
        return

    line_number = 0
    img_genres = []
    with open("dataset.csv", 'r') as file:
        for line in file:
            if line.startswith(movie_slug):
                img_genres = line.split(';')[1]
                break
            line_number += 1

    IMG_DATA, IMG_CLASSES = load_img(line_number, IMAGE_RESOLUTION)
    IMG_DATA = IMG_DATA[0]

    guess = cdouble_to_numpy_array(libc.predict_mlp(model, to_cdouble_array(IMG_DATA), True), 3)

    print(f"Real genres: {img_genres}")
    print(f"Guesses: \nDrama: {guess[0]}\nComedy: {guess[1]}\nAction: {guess[2]}")


def bulk_test(libc, model, nb, imgs, IMAGE_RESOLUTION, TRAINING_PERCENTAGE):
    total_new = [0, 0, 0]
    total_trained = [0, 0, 0]

    guesses_trained = [0, 0, 0]
    guesses_new = [0, 0, 0]

    for i in range(nb):
        print(f"Test : {i}/{nb}")

        new = random.randint(floor(len(imgs) * TRAINING_PERCENTAGE), len(imgs) - 1)
        trained = random.randint(0, floor(len(imgs) - 1 * TRAINING_PERCENTAGE))

        NEW_IMG_DATA, NEW_IMG_CLASSES = load_img(new, IMAGE_RESOLUTION)
        NEW_IMG_DATA = NEW_IMG_DATA[0]
        NEW_IMG_CLASSES = NEW_IMG_CLASSES[0]

        IMG_DATA, IMG_CLASSES = load_img(trained, IMAGE_RESOLUTION)
        IMG_DATA = IMG_DATA[0]
        IMG_CLASSES = IMG_CLASSES[0]

        if NEW_IMG_DATA is None or NEW_IMG_CLASSES is None or IMG_DATA is None or IMG_CLASSES is None:
            i -= 1
            continue

        total_new = [total_new[0] + NEW_IMG_CLASSES[0], total_new[1] + NEW_IMG_CLASSES[1],
                     total_new[2] + NEW_IMG_CLASSES[2]]
        total_trained = [total_trained[0] + IMG_CLASSES[0], total_trained[1] + IMG_CLASSES[1],
                         total_trained[2] + IMG_CLASSES[2]]

        new_guess = cdouble_to_numpy_array(libc.predict_mlp(model, to_cdouble_array(NEW_IMG_DATA), True), 3)
        trained_guess = cdouble_to_numpy_array(libc.predict_mlp(model, to_cdouble_array(IMG_DATA), True), 3)

        guesses_new = [guesses_new[0] + new_guess[0], guesses_new[1] + new_guess[1], guesses_new[2] + new_guess[2]]
        guesses_trained = [guesses_trained[0] + trained_guess[0], guesses_trained[1] + trained_guess[1],
                           guesses_trained[2] + trained_guess[2]]

    # Plt as bars :
    # 4 bars for each classes, 2 for the trained set (real vs guesses), 2 for the new set (real vs guesses)
    # Classes are: 0 : Drama, 1 : Comedy, 2 : Action
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(3)
    opacity = 0.8

    rects1 = plt.bar(index, total_trained, bar_width, alpha=opacity, color='b', label='Real')
    rects2 = plt.bar(index + bar_width, guesses_trained, bar_width, alpha=opacity, color='r', label='Guesses')

    plt.xlabel('Classes')
    plt.ylabel('Quantity')
    plt.title('Trained set')
    plt.xticks(index + bar_width / 2, ('Drama', 'Comedy', 'Action'))
    plt.legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(3)
    opacity = 0.8

    rects1 = plt.bar(index, total_new, bar_width, alpha=opacity, color='b', label='Real')
    rects2 = plt.bar(index + bar_width, guesses_new, bar_width, alpha=opacity, color='r', label='Guesses')

    plt.xlabel('Classes')
    plt.ylabel('Quantity')
    plt.title('New set')
    plt.xticks(index + bar_width / 2, ('Drama', 'Comedy', 'Action'))
    plt.legend()

    plt.tight_layout()
    plt.show()
