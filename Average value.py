import cv2
import numpy as np
import os

dir_path = "images_bpg"

for filename in os.listdir(dir_path):
    if filename.endswith(".png"):
        img_path = os.path.join(dir_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        avg_value = np.average(img)
        print(f"{filename}, {avg_value}")

        # Віднімання середнього значення від кожного пікселя та перетворення від'ємних значень на 0
        #img = np.clip(img - avg_value, 0, None)

        # Збереження нового зображення з добавкою "-AV"
        #new_filename = os.path.splitext(filename)[0] + "-AV.png"
        #new_img_path = os.path.join(dir_path, new_filename)
        #cv2.imwrite(new_img_path, img)