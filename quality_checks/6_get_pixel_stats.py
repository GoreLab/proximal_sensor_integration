import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageSequence
import imageio.v3 as iio
from tqdm import tqdm

dir_path = '/share/data_share/UAV_MSI_data/CV_Experiments/MSI_NYH2_NYH3_2020-24'
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

pixel_stats = pd.DataFrame(np.nan, index=range(len(onlyfiles)), columns=('file_name', 'clip_upper', 'clip_lower', 
                                                                         'NA_count', 'pixel_count', 'clip_ratio', 
                                                                         'NA_ratio', 'im_height', 'im_width'))

for i in tqdm(range(len(onlyfiles))):
    file_name = onlyfiles[i]
    img_pth = f"{dir_path}/{file_name}"
    # for tifs
    img_npy = iio.imread(img_pth) # transparent values set to NA
    #if np.nanmax(img_npy) > 1: print("Clipping upper:", np.sum(img_npy > 1))
    #if np.nanmin(img_npy) < 0: print("Clipping lower:", np.sum(img_npy < 0))
    #if np.isnan(img_npy).sum() != 0: print("NAs:", np.isnan(img_npy).sum())
    pixel_stats.loc[i, 'file_name'] = file_name
    pixel_stats.loc[i, 'clip_upper'] = np.sum(img_npy > 1)
    pixel_stats.loc[i, 'clip_lower'] = np.sum(img_npy < 0)
    pixel_stats.loc[i, 'NA_count'] = np.isnan(img_npy).sum()
    pixel_stats.loc[i, 'pixel_count'] = np.size(img_npy) - np.isnan(img_npy).sum()
    pixel_stats.loc[i, 'clip_ratio'] = np.sum(img_npy > 1)/np.size(img_npy)
    pixel_stats.loc[i , 'NA_ratio'] = np.isnan(img_npy).sum()/np.size(img_npy)
    pixel_stats.loc[i , 'im_height'] = img_npy.shape[0]
    pixel_stats.loc[i , 'im_width'] = img_npy.shape[1]

pixel_stats.to_csv('pixel_stats_NYH2_NYH3_2020-24.csv')

