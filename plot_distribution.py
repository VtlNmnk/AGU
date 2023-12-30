import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

path_to_images = os.path.abspath('images_bpg')
for image in tqdm(os.listdir(path_to_images), desc="images"):
    # if not image.endswith(".bmp"):
    #     continue
    path_to_image = os.path.join(path_to_images, image)
    image_array = cv2.imread(path_to_image, 0)

    hist_dist=plt.hist(image_array, bins=255)
    plt.title("Histogram with 'auto' bins")
    plt.show()
