import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import cv2
from subprocess import call
from sklearn.metrics import mean_squared_error
import sewar
import psnrhvsm as p
from read_csv_plot import run_plot
from plot_grafs import run_plot_fig
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage.util import random_noise



def image_noise(path_to_image, sigma):

    image_array = cv2.imread(path_to_image, 0)
    image_array_noise = random_noise(img_as_float(image_array), mode='gaussian', seed=92,  var=sigma**2)
    image_array_noise = img_as_ubyte(image_array_noise)
    path_to_noised_images = os.path.join("process_images", "noised_" + os.path.basename(path_to_image))
    cv2.imwrite(path_to_noised_images, image_array_noise, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #cv2.imshow('image', image_array_noise)
    #cv2.waitKey(0)
    return path_to_noised_images

def process_image(path_to_image, quantizer_parameter):


    compressed_image_path = path_to_image.replace("noised_", "compressed_")
    compressed_image_path = compressed_image_path.replace(".png", ".jpg")

    coderExt = ".jpg"
    path_to_cod_file = os.path.join(path_to_root_folder, str(tempImagename) + coderExt).replace(" ", "")
    cv2.imwrite(path_to_cod_file, imageArray, [int(cv2.IMWRITE_JPEG_QUALITY), quantizer_parameter])
    cimageArray = cv2.imread(path_to_cod_file, 0)


    call(["bpgenc", "-m 9", "-b 8", f"-q {quantizer_parameter}", "-o", compressed_image_path, path_to_image])

    # call decompressing command
    # usage: bpgdec [options] infile
    # Options:
    # -o outfile.[ppm|png]   set the output filename (default = out.png)
    # -b bit_depth           PNG output only: use bit_depth per component (8 or 16, default = 8)
    # -i                     display information about the image
    decompressed_image_path = path_to_image.replace("noised_", "decompressed_")
    call(["bpgdec", "-o", decompressed_image_path, compressed_image_path])

    return decompressed_image_path, compressed_image_path

def calc_metrics(path_to_image, decompressed_image_path, compressed_image_path, quantizer_parameter):

    init_image = cv2.imread(path_to_image, 0)
    decomp_image = cv2.imread(decompressed_image_path, 0)


    psnrhvsm, psnrhvs, mse_hvs_m, mse_hvs = p.psnrhvsm(init_image, decomp_image)
    mse = mean_squared_error(np.float32(init_image), np.float32(decomp_image))
    psnr = 10 * np.log10(255 * 255 / mse)
    msssim = np.abs(sewar.full_ref.msssim(init_image, decomp_image))

    # check sizes
    source_size = os.path.getsize(path_to_image)
    compress_size = os.path.getsize(compressed_image_path)
    cr = source_size / compress_size
    # print("cr = ", cr)
    # print("psnr = ", psnr)
    # print("quantizer_parameter = ", quantizer_parameter)
    df_local = pd.DataFrame({'image': [os.path.basename(path_to_image).replace("noised_", "")],
                             'mse': [mse],
                             'psnr': [psnr],
                             'mse_hvs': [mse_hvs],
                             'psnrhvs': [psnrhvs],
                             'mse_hvs_m': [mse_hvs_m],
                             'psnrhvsm': [psnrhvsm],
                             'cr': [cr],
                             'msssim': [msssim],
                             'QS': [quantizer_parameter]})

    return df_local

if os.path.isdir("results"):
    print("results folder exists. Please delete it manually!")
    exit()
else:
    os.mkdir("results")

if not os.path.isdir("process_images"):
    os.mkdir("process_images")

path_to_folder_with_images = os.path.abspath('images_bpg')

columns = ['image', 'mse', 'psnr', 'mse_hvs', 'psnrhvs',
                           'mse_hvs_m', 'psnrhvsm', 'cr', 'msssim', 'QS']

quantizer_parameter_array = np.arange(1, 51, 1)
# sigma_array = [0.04]
sigma_array = np.arange(0.0, 0.081, 0.004)
for sigma in tqdm(sigma_array, desc="sigma", leave=True):
    df = pd.DataFrame(columns=columns)
    for quantizer_parameter in tqdm(quantizer_parameter_array, desc="quantizer_parameter"):
        for image_name in tqdm(os.listdir(path_to_folder_with_images), desc="images", position=0, leave=True):

            path_to_image = os.path.join(path_to_folder_with_images, image_name)
            if not image_name.endswith(".png"):
                print(f'File {path_to_image} does not ends with .png')
                image = cv2.imread(path_to_image)

                extention = image_name.split(".")[1]
                new_image_name = image_name.replace(extention, "png")
                new_path_to_image = os.path.join(path_to_folder_with_images, new_image_name)
                if os.path.exists(new_path_to_image):
                    print(f"File {new_path_to_image} exists.")
                else:
                    cv2.imwrite(filename=new_path_to_image, img=image)
                    print(f"File {new_path_to_image} created")
                os.remove(path_to_image)
                path_to_image = new_path_to_image


            path_to_noised_image = image_noise(path_to_image, sigma)
            # print("path_to_noised_image = ", path_to_noised_image)
            decompressed_image_path, compressed_image_path = process_image(path_to_noised_image, quantizer_parameter)
            df_local = calc_metrics(path_to_image, decompressed_image_path, compressed_image_path, quantizer_parameter)
            df = pd.concat([df, df_local], ignore_index=True)

    csv_file_name = os.path.join("results", f"PSNR_noise_noise_{sigma}.csv")
    df.to_csv(csv_file_name, index=False)
    # excel_file_name = os.path.join("results", f"PSNR_noise_noise_{sigma}.xls")
    # df.to_excel(excel_file_name, index=False)
# df_read = pd.read_csv(csv_file_name, usecols=['image', 'QS', 'psnr', 'psnr_noise'])
# lines = plt.plot(df_read.QS, df_read.psnr)
# plt.legend(framealpha=1, frameon=True)
# plt.grid()
# plt.xlabel('QS')
# plt.ylabel('psnr')
# plt.show()

run_plot()
#run_plot_fig()