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

    image_array = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    print("source image array = ", image_array.shape)
    image_array_noise = random_noise(img_as_float(image_array), mode='gaussian', seed=92,  var=sigma**2)
    image_array_noise = img_as_ubyte(image_array_noise)
    path_to_noised_images = os.path.join("process_images", "noised_" + os.path.basename(path_to_image))
    cv2.imwrite(path_to_noised_images, image_array_noise)
    #cv2.imshow('image', image_array_noise)
    #cv2.waitKey(0)
    return path_to_noised_images

def process_image(path_to_image, quantizer_parameter):
    """
    Compress image using bpgenc

    Args:
        path_to_image: full path to image file
        compression_parameter: compression parameter

    Returns:
        cr: compression ratio
    """

    # call compressing command
    # Main options for bpgenc:
    # -h                   show the full help (including the advanced options)
    # -o outfile           set output filename (default = out.bpg)
    # -q qp                set quantizer parameter (smaller gives better quality,
    #                  range: 0-51, default = 28)
    # -f cfmt              set the preferred chroma format (420, 422, 444,
    #                  default=420)
    # -c color_space       set the preferred color space (ycbcr, rgb, ycgco,
    #                  ycbcr_bt709, ycbcr_bt2020, default=ycbcr)
    # -b bit_depth         set the bit depth (8 to 12, default = 8)
    # -lossless            enable lossless mode
    # -e encoder           select the HEVC encoder (jctvc, default = jctvc)
    # -m level             select the compression level (1=fast, 9=slow, default = 8)

    compressed_image_path = path_to_image.replace("noised_", "compressed_")
    compressed_image_path = compressed_image_path.replace(".png", ".bpg")

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

def process_image_agu(path_to_image, quantizer_parameter):
    """
    Compress image using AGU

    Args:
        path_to_image: full path to image file
        compression_parameter: compression parameter

    Returns:
        cr: compression ratio
    """
    # AGU is a high quality DCT 32x32 based lossy image compressor.
    # AGU is FREE for scientific and noncommercial use.
    # It is research version (only for 512x512 grayscale RAW image coding).

    # AGU e <input file> <output file> <Quantization Step> - encode.
    # AGU d <input file> <output file> - decode.
    # AGU w <input file> <output file> - decode without deblocking.
    # AGU p <first RAW file> <second RAW file> - calculation of PSNR.

    # Examples of usage:
    # agu e lenna_t.raw lenna_t.agu 40
    # agu d lenna_t.agu lenna_new.raw
    # agu w lenna_t.agu lenna_new.raw
    # agu p lenna_t.raw lenna_new.raw

    compressed_image_path = path_to_image.replace("noised_", "compressed_")
    compressed_image_path = compressed_image_path.replace(".raw", ".agu")
    print("wine Coders/AGU.EXE", "e", path_to_image, compressed_image_path, quantizer_parameter)
    call(["wine", "Coders/AGU.EXE", "e",  str(path_to_image), str(compressed_image_path), str(quantizer_parameter)])

    # call decompressing command

    decompressed_image_path = path_to_image.replace("noised_", "decompressed_")
    call(["wine", "Coders/AGU.EXE", "d", str(compressed_image_path), str(decompressed_image_path)])

    return decompressed_image_path, compressed_image_path

#def calc_metrics(path_to_image, decompressed_image_path, compressed_image_path, quantizer_parameter):
def calc_metrics(path_to_noised_image, decompressed_image_path, compressed_image_path, quantizer_parameter):
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


def image2raw(image_array, path_to_raw_file):
    image_width, image_height = np.shape(image_array)
    raw_array = np.reshape(image_array, (image_width * image_height, 1), order='C')
    raw_array.tofile(path_to_raw_file)


def raw2image(raw_filename):
    dt = np.dtype(np.uint8)
    image_array = np.fromfile(raw_filename, dtype=dt)
    Nbig = np.shape(image_array)[0]
    N = int(np.sqrt(Nbig))
    image_array = np.reshape(image_array, (N, N), order='C')
    return image_array

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

quantizer_parameter_array = np.arange(1, 50, 2)
# quantizer_parameter_array = [1]
sigma_array = [0.02]
#sigma_array = np.arange(0.0, 0.081, 0.004)
for sigma in tqdm(sigma_array, desc="sigma", leave=True):
    df = pd.DataFrame(columns=columns)
    for quantizer_parameter in tqdm(quantizer_parameter_array, desc="quantizer_parameter"):
        for image_name in tqdm(os.listdir(path_to_folder_with_images), desc="images", position=0, leave=True):

            path_to_image = os.path.join(path_to_folder_with_images, image_name)
            path_to_noised_image = image_noise(path_to_image, sigma)
            print(f"path_to_noised_image = {path_to_noised_image}")

            if not path_to_noised_image.endswith(".raw"):
                print(f'File {path_to_noised_image} does not ends with .raw')
                image = cv2.imread(path_to_noised_image, cv2.IMREAD_GRAYSCALE)
                # if len(image.shape) == 3:
                #     print(f"Image is not grayscale.")
                #     print(f"image.shape before= {image.shape}")
                #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #     print(f"image.shape after= {gray.shape}")
                #     print(f"path_to_noised_image = {path_to_noised_image}")
                #     print("####################################")
                #     cv2.imwrite(filename=path_to_noised_image, img=gray)
                #     print(f"Converted to grayscale.")
                #     image = cv2.imread(path_to_noised_image, cv2.IMREAD_GRAYSCALE)
                print(f"image.shape = {image.shape}")
                if image.shape != (512, 512):
                    print("Image is not 512x512")
                    image = cv2.resize(image, (512, 512))
                    cv2.imwrite(path_to_noised_image, image)
                    image = cv2.imread(path_to_noised_image, cv2.IMREAD_GRAYSCALE)
                    print("Converted to 512x512")
                extention = path_to_noised_image.split(".")[1]
                noised_image_basename = os.path.basename(path_to_noised_image)
                new_noised_image_name = noised_image_basename.replace(f".{extention}", ".raw")
                path_to_raw_noised_image = os.path.join("process_images", new_noised_image_name)
                print(f"path_to_raw_noised_image = {path_to_raw_noised_image}")
                image2raw(image_array=image, path_to_raw_file=path_to_raw_noised_image)
                path_to_original_noised_image = path_to_noised_image
                path_to_noised_image = path_to_raw_noised_image

            print("path_to_noised_image = ", path_to_noised_image)
            decompressed_raw_image_path, compressed_raw_image_path = process_image_agu(path_to_noised_image, quantizer_parameter)
            if decompressed_raw_image_path.endswith(".raw"):
                image_array = raw2image(decompressed_raw_image_path)
                decompressed_image_path = decompressed_raw_image_path.replace(".raw", f".{extention}")
                cv2.imwrite(filename=decompressed_image_path, img=image_array)

            print(f"decompressed_image_path = {decompressed_image_path}")
            df_local = calc_metrics(path_to_original_noised_image, decompressed_image_path, compressed_raw_image_path, quantizer_parameter)
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