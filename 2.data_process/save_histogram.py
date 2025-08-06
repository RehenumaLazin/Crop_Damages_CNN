# Refactored Data Fetching and Preprocessing for Crop Damage Study

import os
import math
import numpy as np
from osgeo import gdal
from joblib import Parallel, delayed

# ---------------------------
# Configuration Paths
# ---------------------------

data_paths = {
    'tmax': r'/GridMet_Data/tmmx',
    'tmin': r'/GridMet_Data/tmmn',
    'sw': r'/GridMet_Data/srad',
    'prec': r'/GridMet_Data/pr',
    'humidity': r'/GridMet_Data/sph',
    'hand': r'/HAND'
}

output_dirs = {
    'original': r'Img_Output',
    'zoomed': r'Img_Output_Zoom'
}

# ---------------------------
# Constants
# ---------------------------

HAND_RANGE = (0.0, 459.415)
BIN_SEQ = np.linspace(0, 5000, 33)
CROP_LOSS_CSV = 'CropLoss_Corn_Flood_May_June.csv'
CLEAN_CSV = 'CropLoss_Corn_Flood_May_June_checked.csv'

# ---------------------------
# Helper Functions
# ---------------------------

def load_image(path):
    return np.transpose(gdal.Open(path).ReadAsArray(), (1, 2, 0)).astype(np.float32)

def normalize(img, min_val, max_val):
    img = (img - min_val) * 5000 / (max_val - min_val)
    return np.clip(img, 0, 5000)

def filter_timespan(img, start_day, end_day, bands):
    start_idx = int(math.floor(start_day / 3)) * bands
    end_idx = int(math.floor(end_day / 3)) * bands
    if end_idx > img.shape[2]:
        pad = np.zeros((img.shape[0], img.shape[1], end_idx - img.shape[2]))
        img = np.concatenate((img, pad), axis=2)
    return img[:, :, start_idx:end_idx]

def calc_histogram(img, bin_seq, bins, times, bands):
    hist = np.zeros((bins, times, bands))
    for i in range(img.shape[2]):
        density, _ = np.histogram(img[:, :, i], bin_seq, density=False)
        if density.sum() > 0:
            hist[:, i // bands, i % bands] = density / density.sum()
    return hist

def calc_hand_histogram(img, bin_seq, bins, times):
    hist = np.zeros((bins, times, 1))
    density, _ = np.histogram(img[:, :, 0], bin_seq, density=False)
    for i in range(times):
        if density.sum() > 0:
            hist[:, i, 0] = density / density.sum()
    return hist

def extract_location(locations, loc1, loc2):
    idx = np.where(np.all(locations[:, :2].astype(int) == [loc1, loc2], axis=1))
    return locations[idx][0][2], locations[idx][0][3]  # state, county

# ---------------------------
# Main Data Class
# ---------------------------

class CropDataProcessor:
    def __init__(self):
        self.data_yield = self._check_and_clean_data()
        self.locations = np.genfromtxt('locations_final_Selected.csv', delimiter=',')
        self.index_all = np.arange(self.data_yield.shape[0])

    def _check_and_clean_data(self):
        data = np.genfromtxt(CROP_LOSS_CSV, delimiter=',')
        missing = [
            i for i in range(data.shape[0])
            if not os.path.isfile(os.path.join(output_dirs['zoomed'], f"{int(data[i, 0])}_{int(data[i, 1])}_{int(data[i, 2])}.npy"))
        ]
        clean_data = np.delete(data, missing, axis=0)
        np.savetxt(CLEAN_CSV, clean_data.astype(int), fmt='%d', delimiter=',')
        return clean_data

    def save_processed_data(self):
        num_samples = self.data_yield.shape[0]
        output_image = np.zeros((num_samples, 32, 90, 6)) # 32 bins, 90 time steps at 3 days interval from May to November, and 6 feature variables
        output_area = np.zeros((num_samples,))
        output_year = np.zeros((num_samples,))
        output_locations = np.zeros((num_samples, 2))
        output_index = np.zeros((num_samples, 2))

        for i, row in enumerate(self.data_yield):
            year, loc1, loc2 = map(int, row[:3])
            filename = f"{year}_{loc1}_{loc2}.npy"
            handfile = f"{loc1}_{loc2}.tif"

            hand_img = load_image(os.path.join(data_paths['hand'], handfile))
            hand_img = normalize(hand_img, *HAND_RANGE)
            hand_hist = calc_hand_histogram(hand_img, BIN_SEQ, 32, 90)

            img = np.load(os.path.join(output_dirs['original'], filename))
            img = filter_timespan(img, 121, 334, 5) # from May (121) to November (334) of a year, 5 vars
            img_hist = calc_histogram(img, BIN_SEQ, 32, 90, 5) # 32 bins, 90 time steps at 3 days interval from May to November, and 5 meteorological variables
            merged = np.concatenate((img_hist, hand_hist), axis=2)

            output_image[i] = merged
            output_area[i] = row[3]
            output_year[i] = year
            output_locations[i] = extract_location(self.locations, loc1, loc2)
            output_index[i] = [loc1, loc2]

        max_val = np.max(output_area)
        k = 5000 / (np.log10(2 * max_val) - np.log10(max_val))
        c = -k * np.log10(max_val)

        #Inverse f(n) = 10^((f(n) -c) / k) - MaxValue
        #output_area =np.power(10, (output_area - c)/k)-MaxValue

        np.save('MaxValue.npy', max_val)
        np.save('k.npy', k)
        np.save('c.npy', c)

        output_area = k * np.log10(output_area + max_val) + c

        np.savez(os.path.join(output_dirs['original'], 'histogram_Corn_MN_MO_metForcing_May_June_without_LW_HAND.npz'),
                 output_image=output_image,
                 output_area=output_area,
                 output_year=output_year,
                 output_locations=output_locations,
                 output_index=output_index)
        print('Data saved successfully.')

if __name__ == '__main__':
    processor = CropDataProcessor()
    processor.save_processed_data()
