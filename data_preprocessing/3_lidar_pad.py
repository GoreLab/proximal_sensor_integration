#!/usr/bin/env python
# coding: utf-8

# # LIDAR Data Preprocessing

# ## Imports

# import h5py
import numpy as np

import pandas as pd
from tqdm import tqdm

from glob import glob
from os import path

import cv2

# Load bad pcs
bad_pc = pd.read_csv("/home/eef52/Documents/Autoencoder_UAV/bad_pc_fnames_2020-24.csv")
bad_pc['file_name'] = bad_pc['file_name'].astype(str) + '.png'

# Load metadata file

hmax, wmax = 0,0
for fname in tqdm(glob("/share/data_share/UGV_lidar_data/CV_Experiments/lidar_NYH2_NYH3_2020-24_projected_lidar_denoised/*.png", recursive=True)):
    if "no_flip" in fname:
        continue
    if bad_pc['file_name'].str.contains(fname.split("/")[6]).any():
        continue
    img = cv2.imread(fname)
    # cut down vertical dimension after denoising and projection
    start = np.where(img.sum(axis=1) > 0)[0][0]
    img = img[start:]
    hmax,wmax=max(hmax, img.shape[0]),max(wmax, img.shape[1])
    if img.shape[0] == hmax:
        fmax=fname

print(hmax, wmax)
hmax += 32 - (hmax%32)
wmax += 32 - (wmax%32)

print(fmax, hmax, wmax)


dil_k = 9
for fname in tqdm(glob("/share/data_share/UGV_lidar_data/CV_Experiments/lidar_NYH2_NYH3_2020-24_projected_lidar_denoised/*.png", recursive=True)):
    if "no_flip" in fname:
        continue

    if bad_pc['file_name'].str.contains(fname.split("/")[6]).any():
        continue

    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    start = np.where(img.sum(axis=1) > 0)[0][0]
    img = img[start:]

    h,w=img.shape[:2]

    if img.max() > 240: # has artifact
        dil = cv2.dilate((img > 240).astype(np.uint8), kernel=np.ones((dil_k, dil_k)).astype(np.uint8))
        img[dil.astype(bool)] = 0

    if h < hmax:
        img = np.concatenate([np.zeros((hmax - h, w)), img], axis=0)

    if w < wmax:
        diff = (wmax - w) // 2
        odd = (wmax - w) % 2
        img = np.concatenate([np.zeros((hmax, diff)), img, np.zeros((hmax, diff + odd))], axis=1)

    cv2.imwrite("/share/data_share/UGV_lidar_data/CV_Experiments/lidar_NYH2_NYH3_2020-24_model_data/" + fname.split("/")[-1], img)
