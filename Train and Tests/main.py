from importlib import reload
from load_dataset import *
from train_model import *
from test_model import *

RGB = True
IMAGE_RESOLUTION = 50 * 74
TRAINING_PERCENTAGE = 0.8
if RGB:
    IMAGE_RESOLUTION *= 3

imgs = index_images()
print("Images indexed")

library_path = "./MLCore.dll"
if os.path.exists(library_path):
    libc = cdll.LoadLibrary(library_path)
else:
    raise Exception(f"The specified library does not exist: {library_path}")

print("Library loaded")

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
    LAYERS = [IMAGE_RESOLUTION, IMAGE_RESOLUTION, 5000, 1000, 3]
    model = libc.create_mlp(to_cint_array(LAYERS), len(LAYERS), 1)

print("Model loaded")

quit = False
while not quit:
    print("Choose an option:")
    print("1. Train model")
    print("2. Test model")
    print("3. Bulk test")
    print("4. Quit")
    option = input("Option: ")

    try:
        if option == "1":
            epochs = input("Enter the number of epochs: ")
            save_rate = input("Enter the save rate: ")
            training_rate = input("Enter the training rate: ")
            train_model(libc, model, IMAGE_RESOLUTION, imgs, TRAINING_PERCENTAGE, epochs, save_rate, training_rate)
        elif option == "2":
            movie_slug = input("Enter the movie slug (anme in the resized_images folder): ")
            test_model(libc, model, IMAGE_RESOLUTION, imgs, movie_slug)
        elif option == "3":
            quantity = input("Enter the number of tests: ")
            bulk_test(libc, model, quantity, imgs, IMAGE_RESOLUTION, TRAINING_PERCENTAGE)
        elif option == "4":
            quit = True
        else:
            print("Invalid option")
    except Exception as e:
        print(f"An error occurred: {e}")

libc.Quit()
