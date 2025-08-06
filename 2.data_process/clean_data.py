# Cleaned and Modular Version of the Code
import os
import math
import numpy as np
from osgeo import gdal
from scipy.ndimage import zoom
from joblib import Parallel, delayed

# ------------------------
# Configuration
# ------------------------

ROOT_DIR = 'GridMet_Data'
ROOT_CROPMASK = 'cropmask'
ROOT_HAND = 'HAND'

OUTPUT_DIR = 'Img_Output'
OUTPUT_ZOOM_DIR = 'Img_Output_Zoom'

# Value Ranges for Normalization
RANGES = {
    'prec': (0.1, 379.8),
    'humidity': (0.000225, 0.02205),
    'tmax': (248.575, 317.525),
    'tmin': (232.65, 301.9),
    'sw': (12.725, 381.925),
    'hand': (0.0, 459.415)
}


# ------------------------
# Utility Functions
# ------------------------

def normalize(img, vmin, vmax):
    img = (img - vmin) * 5000 / (vmax - vmin)
    return np.clip(img, 0, 5000)

def divide_image(img, first, step, num):
    return [img[:, :, first + i * step: first + (i + 1) * step] for i in range(num - 1)]

def merge_image(*img_lists):
    return [np.concatenate(channels, axis=2) for channels in zip(*img_lists)]

def apply_mask(images, masks):
    return [img * np.tile(mask, (1, 1, img.shape[2])) for img, mask in zip(images, masks)]

def read_gdal_image(path):
    return np.transpose(gdal.Open(path).ReadAsArray(), (1, 2, 0)).astype('float32')

def read_and_normalize(path, var):
    img = read_gdal_image(path)
    img[img == -9999] = np.nan
    return normalize(img, *RANGES[var])


def save_images(img, zoomed_img, year, loc1, loc2):
    name = f"{year}_{loc1}_{loc2}.npy"
    np.save(os.path.join(OUTPUT_DIR, name), img)
    np.save(os.path.join(OUTPUT_ZOOM_DIR, name), zoomed_img)
    print(f"Saved: {name}")


# ------------------------
# Main Preprocessing Logic
# ------------------------

def preprocess_file(file):
    if not file.endswith(".tif"):
        return

    raw = file.replace('_', ' ').replace('.', ' ').split()
    loc1, loc2 = map(int, raw[:2])

    for year in range(2007, 2021):
        paths = {
            'tmax': os.path.join(ROOT_DIR, 'tmmx',  str(year), f"{year}_Tmax_3Day", file),
            'tmin': os.path.join(ROOT_DIR, 'tmmn',  str(year), f"{year}_Tmin_3Day", file),
            'prec': os.path.join(ROOT_DIR, 'pr',  str(year), f"{year}_Prec_3Day", file),
            'humidity': os.path.join(ROOT_DIR, 'sph',  str(year), f"{year}_HU_3Day", file),
            'sw': os.path.join(ROOT_DIR, 'srad', str(year), f"{year}_SW_3Day", file),
            'hand': os.path.join(ROOT_HAND, file),
            'mask': os.path.join(ROOT_CROPMASK, str(year), f"{year}_Cropmask", file)
        }

        try:
            images = {var: read_and_normalize(paths[var], var) for var in RANGES}
            mask = gdal.Open(paths['mask']).ReadAsArray().astype(np.uint8)
            mask = np.expand_dims((mask == 1).astype(np.uint8), axis=2)
        except Exception as e:
            print(f"Skipping {file} in {year}: {e}")
            continue

        # Divide each into two yearly chunks
        img_lists = {
            var: divide_image(images[var], 0, 122, 2)
            for var in ['prec', 'humidity', 'tmax', 'tmin', 'sw', 'hand']
        }
        mask_list = divide_image(mask, 0, 1, 2)

        merged = merge_image(img_lists['prec'], img_lists['humidity'], img_lists['tmax'],
                             img_lists['tmin'], img_lists['sw'], img_lists['hand'])
        masked = apply_mask(merged, mask_list)

        for i, img in enumerate(masked):
            current_year = year + i
            zoomed = zoom(img, (48 / img.shape[0], 48 / img.shape[1], 1))
            save_images(img, zoomed, current_year, loc1, loc2)


# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    for _, _, files in os.walk(ROOT_HAND):
        Parallel(n_jobs=12)(delayed(preprocess_file)(file) for file in files)
