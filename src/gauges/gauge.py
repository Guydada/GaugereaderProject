import os
import shutil

import typer
import pandas as pd
import numpy as np
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import cv2

import torch

from torch.utils.data import DataLoader

import src.model.dataset_class as img_dataset

import src.calibrator.app as calibrator

import src.utils.convert_xml as xmlr
import src.utils.envconfig as env
import src.utils.image_editing as ie


class Gauge:
    def __init__(self,
                 timestamp: str,
                 camera_id: str,
                 index: str,
                 description: str,
                 gauge_type: str,
                 ui_calibration: bool = True,
                 calibration_image: str = None,
                 calibration_file: str = None):

        # Outer variables
        self.timestamp = timestamp  # Time of creation
        self.camera_id = camera_id
        self.index = index
        self.description = description
        self.gauge_type = gauge_type
        if gauge_type not in env.GAUGE_TYPES:
            raise ValueError(f'Gauge type {gauge_type} not supported')
        self.calibration_image = calibration_image
        self.calibration = None

        # Inner variables
        self.directory, self.xml_file = env.dir_file_from_camera_gauge(camera_id, index)
        self._init_directory()
        self.calibrated = False

        # Calibration
        if calibration_file is not None:  # TODO: Parse XML file to get the same dictionary
            self.calibrate_from_xml()
        elif ui_calibration:  # TODO: prompt for calibration if given
            self.ui_calibrate()
        else:
            raise ValueError('No calibration data given and no UI calibration requested')

    def _init_directory(self):  # TODO: add overwrite/skip option
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            typer.echo(f'Created directory {self.directory}')

    def ui_calibrate(self) -> None:
        ca = calibrator.Calibrator(calibration_image=self.calibration_image,
                                   index=self.index,
                                   camera_id=self.camera_id,
                                   gauge_type=self.gauge_type)
        self.calibration = ca.run()
        self.calibrated = True
        return None

    def calibrate_from_xml(self):
        """
        Calibrate the gauge using the calibration data, save the calibration data to the xml file
        :return:
        """
        self.calibration = xmlr.xml_to_dict(self.xml_file, gauge=True)
        self.calibrated = True
        return None

    def create_train_test_set(self,
                              train_size: float = 0.8,
                              test_size: float = 0.2,
                              random_state: int = 42):
        pass

    def save(self):
        pass

    @classmethod
    def load(cls,
             path: str):
        pass

    def as_dict(self):
        dic = {'gauge_id': self.index,
               'calibration_image': self.calibration_image,
               'calibration_file': self.xml_file}
        return dic


class AnalogGauge(Gauge):
    def __init__(self,
                 timestamp: str,
                 camera_id: str,
                 index: str,
                 description: str,
                 ui_calibration: bool = True,
                 calibration_image: str = None,
                 calibration_file: str = None):
        super().__init__(timestamp=timestamp,
                         camera_id=camera_id,
                         index=index,
                         description=description,
                         gauge_type='analog',
                         ui_calibration=ui_calibration,
                         calibration_image=calibration_image,
                         calibration_file=calibration_file)
        self.train_image_path = os.path.join(self.directory, env.TRAIN_IMAGE_NAME)
        self.train_image = None
        self.needle_image_path = os.path.join(self.directory, env.NEEDLE_IMAGE_NAME)
        self.needle_image = None
        self.data_cols = ['image_name', 'augmented', 'angle']
        self.train_df = pd.DataFrame(columns=self.data_cols)
        self.test_df = pd.DataFrame(columns=self.data_cols)
        self.train_images_path = os.path.join(self.directory, 'train')
        self.test_images_path = os.path.join(self.directory, 'test')

        # Model variables and data structures
        self.trained = False
        self.scores = None
        self.train_image_set = None
        self.test_image_set = None

    def create_train_test_set(self,
                              train_size: float = 0.8,
                              test_size: float = 0.2,
                              random_state: int = 42):
        for path in [self.train_images_path, self.test_images_path]:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                check = typer.prompt(f'Directory "{os.path.basename(path)}" already exists. Overwrite?',
                                     default=True,
                                     show_choices=True)
                if check:
                    shutil.rmtree(path)
                    os.makedirs(path)
        if not self.calibrated:
            raise ValueError('Gauge not calibrated')
        self.train_image = cv2.imread(self.train_image_path)
        if self.train_image is None:
            raise FileNotFoundError(f'Train image "{self.train_image}" not found')
        self.needle_image = cv2.imread(self.needle_image_path)
        if self.needle_image is None:
            raise FileNotFoundError(f'Needle image "{self.needle_image}" not found')
        min_angle = float(self.calibration['min_angle'])
        max_angle = float(self.calibration['max_angle'])
        center = self.calibration['center']
        # convert values in tuple to floats
        center = tuple([float(x) for x in center])
        set_size = env.IMAGE_TRAIN_SET_SIZE
        train_angles = np.linspace(min_angle, max_angle, set_size)
        test_angles = np.random.uniform(min_angle, max_angle, env.IMAGE_TEST_SET_SIZE)
        self.create_set(angle_list=train_angles,
                        set_type='train',
                        center=center)
        self.train_df.to_csv(os.path.join(self.directory, 'train.csv'), index=False)
        typer.secho(f'Train set created, contains: {len(self.train_df)} images', fg='green')
        self.create_set(angle_list=test_angles,
                        set_type='test',
                        center=center)
        self.test_df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)
        typer.secho(f'Test set created, contains: {len(self.test_df)} images', fg='green')
        return None

    def create_set(self,
                   angle_list: np.array,
                   set_type: str,
                   center: tuple):
        if set_type not in ['train', 'test']:
            raise ValueError('Set type must be train or test')
        index = 0
        for angle in angle_list:
            index += 1
            image_name = f'{index:05d}' + '.jpg'
            image = ie.rotate_needle(self.train_image, self.needle_image, center, angle)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil.save(f"{self.directory}/{set_type}/{image_name}")
            if set_type == 'train':
                self.train_df = self.train_df.append(pd.DataFrame([[image_name, False, angle]], columns=self.data_cols))
            else:
                self.test_df = self.test_df.append(pd.DataFrame([[image_name, False, angle]], columns=self.data_cols))
            index += 1
            image_name = f'{index:05d}' + '.jpg'
            image_pil = ie.image_augmentor(image_pil)
            if np.random.randint(0, 2) == 1:
                image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=np.random.randint(1, 3)))
            image_pil.save(f"{self.directory}/{set_type}/{image_name}")
            if set_type == 'train':
                self.train_df = self.train_df.append(pd.DataFrame([[image_name, True, angle]], columns=self.data_cols))
            else:
                self.test_df = self.test_df.append(pd.DataFrame([[image_name, True, angle]], columns=self.data_cols))

    def init_train(self,
                   model_name: str,
                   transform = None,
                   evaluate: bool = False):
        if not self.calibrated:
            raise ValueError('Gauge not calibrated')
        self.train_image_set = img_dataset.ImageDataset(gauge_directory=self.directory,
                                                        set_type='train',
                                                        transform=transform)
        train_data_loader = DataLoader(self.train_image_set,
                                       batch_size=env.BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=env.NUM_WORKERS)
