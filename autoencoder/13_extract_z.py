import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
from dataset import MyDataset, ExtractZ
import torch
from torch import Tensor
from torchvision import transforms
import pandas as pd


#--------------------------EXTRACT Z FROM TRAINED AUTOENCODER MODEL-------------------------------

def extract_z(params: dict, train_transforms: list):
    # Get extraction and AE params
    config_extract_z = params
    with open(config_extract_z['AE_model_params']['config_path'], 'r') as file:
        try:
            config_AE = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # Initialize best experiment & load checkpoint
    model = vae_models[config_AE['model_params']['name']](**config_AE['model_params'])
    best_experiment = VAEXperiment(model,
                          config_AE['exp_params'])
    checkpoint = torch.load(config_extract_z['AE_model_params']['checkpoint_path'])
    best_experiment.load_state_dict(checkpoint["state_dict"])
    best_experiment.eval()
    # Load image data
    dataset = ExtractZ(
             annotations_file = config_extract_z['AE_data_params']['labels_path'],
             file_name = config_extract_z['AE_data_params']['file_name_column'],
             data_path = config_extract_z['AE_data_params']['data_path'],
             data_type = config_extract_z['AE_data_params']['data_type'],
             transform = train_transforms,
             target_transform = None
         )
    # Extract z
    z_all = torch.empty(len(dataset), config_extract_z['AE_data_params']['out_channels'])
    for i in range(len(dataset)):
        with torch.no_grad():
            z = best_experiment.model.encoder(torch.unsqueeze(dataset[i][0], 0)).cpu()
        z = torch.mean(z, dim = 3)
        z = torch.mean(z, dim = 2)
        z_all[i] = z.squeeze()
    return z_all


#------------------RUN FUNCTION TO EXTRACT LSPS------------------------------

# with open('configs/extract_z_lidar.yaml', 'r') as file:
#     try:
#         config_extract_z_lidar = yaml.safe_load(file)
#     except yaml.YAMLError as exc:
#         print(exc)

with open('configs/extract_z_msi.yaml', 'r') as file:
    try:
        config_extract_z_msi = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# transforms_lidar = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.5], std=[0.5]),
#                                         transforms.GaussianBlur(kernel_size=(5, 9), sigma=1),
#                                         ])

transforms_msi = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop((384,96))])

# z_lidar = extract_z(config_extract_z_lidar, transforms_lidar)
# z_lidar_np = z_lidar.numpy()
# np.savetxt('lsp_files/z_lidar_Test_512.csv', z_lidar_np, delimiter = ',')

z_msi = extract_z(config_extract_z_msi, transforms_msi)
z_msi_np = z_msi.numpy()
np.savetxt('lsp_files/z_msi_Test_512.csv', z_msi_np, delimiter = ',')


