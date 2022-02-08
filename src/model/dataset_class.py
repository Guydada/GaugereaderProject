import os
import cv2
import torch
import pandas as pd
import skimage.io as io

from torch.utils.data import Dataset

import src.utils.image_editing as ie


import src.utils.envconfig as env


class ImageDataset(Dataset):
    def __init__(self,
                 gauge_directory: str,
                 set_type: str,
                 transform = None):
        # Outer variables
        self.gauge_directory = gauge_directory
        self.set_type = set_type
        self.transform = transform
        if set_type not in ['train', 'val', 'test']:
            raise ValueError('set_type must be one of train, val, test')

        # Paths
        self.images_path = os.path.join(self.gauge_directory, set_type)

        # Data
        self.set_df = pd.read_csv(os.path.join(self.gauge_directory, '{}.csv'.format(set_type)))
        self.image_name_frame = self.set_df['image_name']
        self.angles = self.set_df['label'].values

    def __getitem__(self, index):
        image_name = os.path.join(self.images_path,
                                  self.image_name_frame[index])
        image = io.imread(image_name)
        angle = self.angles[index]
        if self.transform is not None:
            self.transform(image, angle)
        return image, angle

    def __len__(self):
        return len(self.set_df)