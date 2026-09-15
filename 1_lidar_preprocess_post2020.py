#!/usr/bin/env python
# coding: utf-8

# # LIDAR Data Preprocessing

# ## Imports

import laspy
import numpy as np

import pandas as pd
from tqdm import tqdm

from glob import glob
from os import path

import cv2

out_dir = "/share/data_share/UGV_lidar_data/2021/2021_NYH2_projected_lidar_denoised"

base_pc_pth = "/share/data_share/UGV_lidar_data/LidarPointClouds_2021_2022/AzureStorage/outputs/point-cloud/"
base_md_pth = "/share/data_share/UGV_lidar_data/PlotSplits_2021/"

md_files = [
    "NYH2_2021-06-25.csv",
    "NYH2_2021-07-01.csv",
    "NYH2_2021-07-10.csv",
    "NYH2_2021-07-15.csv",
    "NYH2_2021-07-16.csv",
    "NYH2_2021-07-22.csv",
    "NYH2_2021-07-27.csv",
    "NYH2_2021-08-03.csv",
    "NYH2_2021-08-10.csv",
    "NYH2_2021-10-13.csv"
]

# noise filter
# radial

# need to stratify into train/test
with open("./not_in_md.txt", "w") as not_in_md, open("./bad_pc.txt", "w") as bad_pc:
    for md_f in md_files[::-1]:
        # Load metadata file
        md = pd.read_csv(path.join(base_md_pth, md_f))

        for _, row in tqdm(md.iterrows()):
            fname = row["Timestamp"].split("T")[0] + "_" + row['Collection ID'] + "_range_" + str(row['Range'])
            out_pth = path.join(out_dir, fname)

            if path.exists(out_pth + ".png"):
                continue

            try:
                with laspy.open(path.join(base_pc_pth, row['Collection ID'], "range_"+str(row['Range']), "point-cloud.las")) as f:
                    pts_f = f.read().points
                    pts_f = np.array([pts_f.x, pts_f.y, pts_f.z]).T
            except:
                bad_pc.writelines([path.join(base_pc_pth, row['Collection ID'], "range_"+str(row['Range'])) + " doesn't exist\n"])
                continue

            if row['Direction'] == -1: # rover approaching from opposite direction, need to compensate
                pts_f[:, 0] = -pts_f[:, 0] # negate x (left becomes right and vice versa)
                pts_f[:, 1] = pts_f[:, 1].max() - pts_f[:, 1]

            # Remove points from adjacent plots
            pts_f = pts_f[(pts_f[:, 0] > -0.762) & (pts_f[:, 0] < 0.762)]
            # Remove Ground Points
            pts_f = pts_f[pts_f[:, 2] > 0.]

            # ## Voxelize & Denoise
            try:
                min_pts_f, max_pts_f = pts_f.min(axis=0), pts_f.max(axis=0)
            except:
                bad_pc.writelines([fname + '\n'])
                continue

            voxel_res = 5e-3 # 5mm lateral and vertical grid resolution (XZ)
            num_voxels = 1 + np.ceil((max_pts_f - min_pts_f) / voxel_res).astype(int)

            time = sorted(np.unique(pts_f[:,1])) #densely sampled data also removed
            T = len(time)
            if T > 500:
                bad_pc.writelines([fname + " not split\n"])

            voxel_grid = np.zeros((num_voxels[0], T, num_voxels[2]))
            #print(voxel_grid.shape)

            coords = np.round((pts_f - min_pts_f) / voxel_res).astype(int)
            coords[:, 1] = np.nonzero(pts_f[:, 1, None] == time)[1]
            uniq, counts = np.unique(coords, axis=0, return_counts=True)
            voxel_grid[uniq[:, 0], uniq[:, 1], uniq[:, 2]] = counts

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
                bad_pc.writelines([fname + " no points\n"])
            else:
                cv2.imwrite(path.join(out_dir, out_pth+".png"), density.clip(0, 255))
                cv2.imwrite(path.join(out_dir, out_pth+"_no_flip.png"), density_no_flip.clip(0, 255))