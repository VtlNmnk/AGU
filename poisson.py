import numpy as np
import os
import cv2
def image_noise(path_to_image):
    image_array = cv2.imread(path_to_image, 0)

    np.random.seed(0)
    poisson_noise = np.sqrt(image_array) * np.random.normal(0, 1, image_array.shape)
    noisy_image = image_array + poisson_noise

    path_to_noised_images = os.path.join("process_images", "noised_" + os.path.basename(path_to_image))
    cv2.imwrite(path_to_noised_images, noisy_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    return path_to_noised_images