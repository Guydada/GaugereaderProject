import os
import shutil
import typer
import cv2
import pandas as pd
import numpy as np
import skimage.io as io
import PIL.Image as Image

from torch.utils.data import Dataset

import src.utils.image_editing as ie
from config import settings


class ImageDataset(Dataset):
    def __init__(self,
                 set_type: str,
                 calibration: dict,
                 transform=ie.process_image):
        # Outer variables
        self.set_type = set_type
        if set_type not in ['train', 'val', 'test']:
            raise ValueError('set_type must be one of train, val, test')
        self.calibration = calibration
        self.transform = transform
        directory = calibration['directory']
        # Paths
        self.images_path = os.path.join(directory, set_type)
        self.gauge_directory = self.calibration['directory']
        self.report_path = os.path.join(self.gauge_directory, f'{self.set_type}_df.csv')

    def initialize_dir(self):
        """
        Initialize the dataset's directory
        """
        if os.path.exists(self.images_path):
            typer.secho('Images directory already exists, overwriting', fg='yellow')
            shutil.rmtree(self.images_path)
        os.makedirs(self.images_path)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.set_df)


class AnalogDataSet(ImageDataset):
    def __init__(self,
                 set_type: str,
                 calibration: dict,
                 base_image: np.ndarray,
                 needle_image: np.ndarray,
                 angles: np.ndarray,
                 transform=ie.process_image):
        super().__init__(set_type=set_type,
                         calibration=calibration,
                         transform=transform)
        # Outer variables
        self.base_image = base_image
        self.needle_image = needle_image
        self.angles = angles
        # Inner variables
        self.data_cols = ['image_name', 'augmented', 'real_angle', 'radians']
        center = self.calibration['center']
        self.center = tuple([float(x) for x in center])
        try:
            self.set_df = pd.read_csv(os.path.join(self.gauge_directory, f'{self.set_type}_df.csv'))
        except FileNotFoundError:
            self.set_df = self.create_dataset()

    def create_dataset(self):
        """
        Creates the synthetic dataset from base image and needle image, angle list
        :return:
        """
        self.set_df = pd.DataFrame(columns=self.data_cols)
        self.initialize_dir()
        for index, angle in enumerate(self.angles, start=1):
            image, _ = ie.rotate_needle(self.base_image, self.needle_image, self.center, angle)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_name = f'{index:05d}.jpg'
            image = cv2.resize(image, settings.TRAIN_IMAGE_SHAPE)
            cv2.imwrite(os.path.join(self.images_path, image_name), image)
            self.set_df = pd.concat([self.set_df,
                                     pd.DataFrame([[image_name, False, angle, angle]],
                                                  columns=self.data_cols)])
        self.set_df['radians'] = self.set_df['radians'].apply(lambda x: np.radians(x))
        self.set_df.to_csv(self.report_path, index=False)
        return self.set_df

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path,
                                  self.set_df.iloc[index]['image_name'])
        image = io.imread(image_path)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.set_df.iloc[index]['radians']

    def __len__(self):
        return len(self.set_df)
