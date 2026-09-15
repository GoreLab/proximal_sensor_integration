import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import v2
import zipfile
from torch.utils.data import random_split
from torchvision.transforms import functional as F

import numpy as np
from PIL import Image, ImageSequence
import random
from torchvision.transforms import Lambda
import pandas as pd
import torchvision.datasets as dset
from torch.utils.data import ConcatDataset
import math
import imageio.v3 as iio

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 data_type: str,
                 annotations_file: str,
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path)    
        self.data_type = data_type   
        self.transforms = transform
        self.img_labels = pd.read_csv(annotations_file)
        imgs = [f for f in self.data_dir.iterdir() if f.suffix == '.png' or f.suffix == '.tif']
        random.shuffle(imgs)
        self.imgs = []
        if split == "train":
            for img in imgs:
                record = self.img_labels.index[self.img_labels['file_name'] == img.name].tolist()
                if len(record) == 1 and self.img_labels.iloc[record[0]]['train_set'] == 'train':
                    self.imgs.append(img)
        else:
            for img in imgs:
                record = self.img_labels.index[self.img_labels['file_name'] == img.name].tolist()
                if len(record) == 1 and self.img_labels.iloc[record[0]]['train_set'] == 'val':
                    self.imgs.append(img)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if self.data_type == 'msi':
            img_npy = iio.imread(img_path)
        if self.data_type == 'lidar':
            img_pil = Image.open(img_path) # load PIL image object. if you just use this it'll be a single channel
            img_npy = np.empty((*img_pil.size[::-1], 0), dtype=np.uint8) # this is where we'll store all 5 channels
            for im in ImageSequence.Iterator(img_pil): # iterate through 5 channels
                img_npy = np.concatenate([img_npy, np.array(im)[..., None]], axis=-1) # add it to our numpy array
        if self.transforms is not None:
            if self.data_type == "msi":
                img = self.transforms(img_npy) # since we are feeding in a npy array now instead of PIL moved ToTensor() up front. since the other transforms need PIL or torch
                mean = torch.mean(img)
                img = img - mean
            if self.data_type == 'lidar':
                img = self.transforms(img_npy)
        return img, 0.0 # dummy datat to prevent breaking


#Load all images and labels; unshuffled; unsplit; use after model is trained to pass all data through and extract latent codes (z)
class ExtractZ(Dataset):
    def __init__(self,
                 annotations_file: str,
                 file_name: str,
                 data_path: str,
                 data_type: str,
                 transform: Callable,
                 target_transform: Callable,
                **kwargs):
        self.img_labels = pd.read_csv(annotations_file)
        self.file_name = file_name
        self.data_dir = Path(data_path) 
        self.data_type = data_type      
        self.transforms = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_labels[self.file_name].iloc[idx])
        if self.data_type == 'msi':
            img_npy = iio.imread(img_path)
        if self.data_type == 'lidar':
            img_pil = Image.open(img_path) # load PIL image object. if you just use this it'll be a single channel
            img_npy = np.empty((*img_pil.size[::-1], 0), dtype=np.uint8) # this is where we'll store all 5 channels
            for im in ImageSequence.Iterator(img_pil): # iterate through 5 channels
                img_npy = np.concatenate([img_npy, np.array(im)[..., None]], axis=-1) # add it to our numpy array
        if self.transforms is not None:
            if self.data_type == 'msi':
                img = self.transforms(img_npy) # since we are feeding in a npy array now instead of PIL moved ToTensor() up front. since the other transforms need PIL or torch
                mean = torch.mean(img)
                img = img - mean
            if self.data_type == 'lidar':
                img = self.transforms(img_npy)
        return img, 0.0 #can only load discrete labels


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        annotations_file: str,
        data_type: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_path
        self.annotations_file = annotations_file
        self.data_type = data_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def setup(self, stage: Optional[str] = None) -> None:
        
#       =========================  My Dataset  =========================

        if self.data_type == 'msi':
            train_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.CenterCrop((384,96)),
                                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(1, 5)),
                                                transforms.RandomHorizontalFlip(p = 0.5),
                                                transforms.RandomVerticalFlip(p= 0.5)]) #All images need to be rescaled to the same size
            val_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.CenterCrop((384,96))]) #All images need to be rescaled to the same size

        if self.data_type == 'lidar':
            train_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5]),
                                                transforms.RandomAffine(degrees = 0, translate = (0.2, 0), fill = -1),
                                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=1),
                                                transforms.RandomHorizontalFlip(p = 0.5)
                                                ]) #All images need to be rescaled to the same size
            val_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5]),
                                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=1)
                                                ]) #All images need to be rescaled to the same size

        target_transforms = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y, dtype = torch.int64), value=1))

        self.train_dataset = MyDataset(
            data_path=self.data_dir,
            data_type=self.data_type,
            annotations_file=self.annotations_file,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        self.val_dataset = MyDataset(
            data_path=self.data_dir,
            data_type=self.data_type,
            annotations_file=self.annotations_file,
            split='val',
            transform=val_transforms,
            download=False,
        )

        #If need to load to dataloader for extracting latent codes (z), use val_dataloader
        self.pred_dataset = ExtractZ(
            annotations_file=self.annotations_file,
            file_name='file_name',
            data_path=self.data_dir,
            data_type=self.data_type,
            transform=train_transforms,
            target_transform=target_transforms,
            download=False,
        )
        
#       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

