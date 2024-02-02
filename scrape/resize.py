# get all images in images folder, crop them to 50x74 and save them in resized_images
import os
from PIL import Image

fmt = (50, 74)

i = 0
imgs = os.listdir('images')
for file in imgs:
    i += 1

    if os.path.isfile(f'resized_images/{file}'):
        continue

    print(f'Resizing: {i} / {imgs.__len__()}')

    img = Image.open(f'images/{file}').convert('RGB')
    img = img.resize(fmt)
    img.save(f'resized_images/{file}')
