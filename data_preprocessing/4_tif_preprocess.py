import numpy as np
from os import listdir
from os.path import isfile, join
import imageio.v3 as iio
import math
from tqdm import tqdm

dir_path = '/share/data_share/UAV_MSI_data/CV_Experiments/MSI_NYH2_NYH3_2020-24'
out_dir = '/share/data_share/UAV_MSI_data/CV_Experiments/MSI_NYH2_NYH3_2020-24_clipped_fillmean'
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

with open("tif_preprocess_fillmean.txt", "w") as f:
    for i in tqdm(range(len(onlyfiles))):
        file_name = onlyfiles[i]
        img_pth = f"{dir_path}/{file_name}"
        # For tifs
        img_npy = iio.imread(img_pth) # transparent values set to NA
        img_npy = np.clip(img_npy, 0, 1) # set values greater than 1 equal to 1

        # If no NaNs, save file and continue
        if np.isnan(img_npy).sum() == 0:
            iio.imwrite(f"{out_dir}/{file_name}", img_npy)
            continue

        # All 8 neighbor directions (up, down, left, right, and diagonals)
        directions = [(-1,  0), (1,  0), (0, -1), (0,  1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Fill each NaN with average of valid neighbors
        filled_arr = img_npy.copy()
        max_iters = 100

        for iteration in range(max_iters):
            changed = False
            new_arr = filled_arr.copy()
            for channel in range(5):
                nan_indices = np.argwhere(np.isnan(filled_arr[:,:,channel]))
                for i, j in nan_indices:
                    neighbor_vals = []
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < filled_arr.shape[0] and 0 <= nj < filled_arr.shape[1]:
                            val = filled_arr[ni, nj, channel]
                            if not math.isnan(val):
                                neighbor_vals.append(val)
                    if neighbor_vals:
                        new_value = np.mean(neighbor_vals)
                        new_arr[i, j, channel] = new_value
                        changed = True
            filled_arr = new_arr
            if not changed:
                f.writelines(f"{file_name} ✅ Converged after {iteration+1} iterations.\n")
                break
            if changed and iteration == max_iters - 1:
                f.writelines(f"{file_name} ⚠️ Maximum iterations reached without full convergence.\n")

        # Save image
        iio.imwrite(f"{out_dir}/{file_name}", filled_arr)
