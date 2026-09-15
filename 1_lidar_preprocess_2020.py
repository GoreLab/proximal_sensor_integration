#!/usr/bin/env python
# coding: utf-8

# # LIDAR Data Preprocessing

# ## Imports

import h5py
import numpy as np

import pandas as pd
from tqdm import tqdm
import json

from glob import glob
from os import path

import cv2

out_dir = "/share/data_share/UGV_lidar_data/2020/2020_projected_lidar_denoised/"

base_pc_pth = "/share/data_share/UGV_lidar_data/LidarPointClouds_2020/"
base_md_pth = "/share/data_share/UGV_lidar_data/"

# Load metadata file
md = pd.read_csv(path.join(base_md_pth, "Plotsplits_2020_unnested.csv"))
md['height_left'] = ''
md['height_right'] = ''
md['height_all'] = ''

not_in_md = open("not_in_md.txt", "w")
bad_pc = open("bad_pc.txt", "w")

for _, row in tqdm(md.iterrows()):
    # Save height info from json file
    with open(path.join(base_pc_pth, row["timestamp"].split("T")[0], row['collection'], "height-500", "range_"+str(row['range']), "height.json"), 'r') as f:
        height = json.load(f)
        md.loc[_, 'height_left'] = height["left"]
        md.loc[_, 'height_right'] = height["right"]
        md.loc[_, 'height_all'] = height["all"]
    
    fname = row["timestamp"].split("T")[0] + "_" + row['collection'] + "_range_" + str(row['range'])
    out_pth = path.join(out_dir, fname)

    if path.exists(out_pth + ".png"):
        continue

    # Import file & convert data to numpy array
    try:
        with h5py.File(path.join(base_pc_pth, row["timestamp"].split("T")[0], row['collection'], "height-500", "range_"+str(row['range']), "point_cloud.h5"), 'r') as f:
            dset = f['cloud']
            pts_f = np.array(dset)
    except:
        bad_pc.writelines([path.join(base_pc_pth, row["timestamp"].split("T")[0], row['collection'], "height-500", "range_"+str(row['range'])) + " doesn't exist\n"])
        continue

    # ## Flip Plots to be in Same Orientation (to account for rover driving pattern)
    if row['start.range'] > row['end.range']: # rover approaching from opposite direction, need to compensate
        pts_f[:, 0] = -pts_f[:, 0] # negate x (left becomes right and vice versa)
        pts_f[:, 1] = pts_f[:, 1].max() - pts_f[:, 1]

    # Remove points from adjacent plots
    pts_f = pts_f[(pts_f[:, 0] > -0.762) & (pts_f[:, 0] < 0.762)]
    # Remove Ground Points
    pts_f = pts_f[pts_f[:, 2] > 0.2]

    # ## Voxelize & Denoise
    try:
        min_pts_f, max_pts_f = pts_f.min(axis=0), pts_f.max(axis=0)
    except:
        bad_pc.writelines([fname + "\n"])
        continue

    voxel_res = 5e-3 # 5mm lateral and vertical grid resolution (XZ)
    num_voxels = 1 + np.ceil((max_pts_f - min_pts_f) / voxel_res).astype(int)

    time = sorted(np.unique(pts_f[:,1])) #densely sampled data also removed
    T = len(time)
    if T > 500:
        bad_pc.writelines([fname + " not split\n"])

    voxel_grid = np.zeros((num_voxels[0], T, num_voxels[2]))
    #print(voxel_grid.shape)

    for pt in pts_f:
        coord = np.round((pt - min_pts_f) / voxel_res).astype(int)
        coord[1] = np.where(pt[1] == time)[0]
        voxel_grid[coord[0], coord[1], coord[2]] += 1

    # As described in paper, if voxel contains single point and no neighbors, remove it.
    voxel_grid_denoised = voxel_grid.copy()
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if voxel_grid[i,j,k] == 1 and voxel_grid[i-1:i+1,j-1:j+1,k-1:k+1].sum() == 1:
                    voxel_grid_denoised[i,j,k] = 0

    # ## Convert to Density Map

    # ### Flip grid vertically
    num_voxels_lateral = voxel_grid_denoised.shape[0]
    if num_voxels_lateral % 2 == 0:
        voxel_grid_flipped = voxel_grid_denoised[:num_voxels_lateral//2] + voxel_grid_denoised[num_voxels_lateral//2:][::-1]
    else:
        voxel_grid_flipped = voxel_grid_denoised[:num_voxels_lateral//2] + voxel_grid_denoised[1+num_voxels_lateral//2:][::-1]

    # ### Project onto vertical-lateral plane
    density = voxel_grid_flipped.sum(axis=1).T[::-1]
    density_no_flip = voxel_grid_denoised.sum(axis=1).T[::-1]

    # cut down vertical dimension after denoising and projection
    start = np.where(density.sum(axis=1) > 0)[0][0]
    density = density[start:]
    density_no_flip = density_no_flip[start:]

    #assert density.max() < 256 and density_no_flip.max() < 256, fname

    # print(path.join("/home/pm568/projects/Autoencoder_UAV/lidar", out_pth+".png"))
    if density.size == 0 or density_no_flip.size == 0:
        bad_pc.writelines([fname])
    else:
        cv2.imwrite(path.join(out_dir, out_pth+".png"), density.clip(0, 255))
        cv2.imwrite(path.join(out_dir, out_pth+"_no_flip.png"), density_no_flip.clip(0, 255))

not_in_md.close()
bad_pc.close()
md.to_csv('/home/eef52/Documents/Autoencoder_UAV/PlotSplits_2020_unnested_height.csv')