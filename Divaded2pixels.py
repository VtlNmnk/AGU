
import os
import cv2

# шлях до папки з зображеннями
dir_path = "images_bpg"

# проходимося по кожному файлу в папці
for filename in os.listdir(dir_path):
    if filename.endswith(".png"):
        # зчитуємо зображення
        img = cv2.imread(os.path.join(dir_path, filename), cv2.IMREAD_GRAYSCALE)

        # ділимо кожен піксель на 2 та округлюємо до цілого
        img_div2 = (img // 2).astype('uint8')

        # зберігаємо нове зображення з добавкою "-DIV2"
        new_filename = os.path.splitext(filename)[0] + "-DIV2.png"
        cv2.imwrite(os.path.join(dir_path, new_filename), img_div2)