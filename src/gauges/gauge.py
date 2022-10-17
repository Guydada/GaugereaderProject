import os
import cv2
import typer
import torch

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

import src.model.dataset_class as img_dataset
import src.model.gauge_net as gn
import src.calibrator.app as calibrator
import src.utils.convert_xml as xmlr
import src.utils.image_editing as ie
import src.utils.envconfig as env

from config import settings


class Gauge:
    def __init__(self,
                 calibration: dict or str = None,
                 transfer_learning: bool = False):
        """
        Initialize the gauge.
        :param calibration: Calibration dictionary or path to the calibration xml file.
        """
        if isinstance(calibration, dict):
            self.calibration = calibration
        elif isinstance(calibration, str):
            path = os.path.join(settings.XML_FILES_PATH, calibration)
            self.calibration = xmlr.xml_to_dict(path)
            self.calibration = self.calibration['gauge']
        # Inner variables
        self.directory = self.calibration['directory']
        if not os.path.exists(self.directory):
            self.directory = env.get_directory(self.calibration['index'], self.calibration['camera_id'])
            self.calibration['directory'] = self.directory
        self.transfer_learning = transfer_learning

    def initialize(self):
        """
        Start the gauge application.
        """
        pass

    def train(self):
        """
        Train the model.
        """
        pass

    def create_train_val_set(self):
        pass

    def visual_test(self,
                    model):
        """
        Test the model's performance.
        """
        pass

    def get_reading(self,
                    model,
                    frame: str or np.ndarray):
        """
        Get the reading of the gauge.
        :param model:
        :param frame:
        :return:
        """
        pass

    @classmethod
    def calibrate(cls,
                  calibration_image: str = None,
                  index: int = 1,
                  camera_id: int = 1):
        """
        Call the Calibration APP for the gauge.
        :param calibration_image:
        :param index: Index of the gauge.
        :param camera_id: Camera ID of the gauge.
        :return: None
        """
        pass


class AnalogGauge(Gauge):
    def __init__(self,
                 calibration: dict or str = None):
        super().__init__(calibration=calibration)
        # Train/test set directories
        self.train_image_path = os.path.join(self.directory, settings.TRAIN_IMAGE_NAME)
        self.needle_image_path = os.path.join(self.directory, settings.NEEDLE_IMAGE_NAME)

        # Train/test base images
        self.base_image = cv2.imread(self.train_image_path)
        if self.base_image is None:
            raise FileNotFoundError(f'Train image "{self.base_image}" not found')
        self.needle_image = cv2.imread(self.needle_image_path)
        if self.needle_image is None:
            raise FileNotFoundError(f'Needle image "{self.needle_image}" not found')

        self.angles = None
        self.datasets = None
        self.data_loaders = None
        self.model = None

    def initialize(self,
                   force_train: bool = False):
        # Angles
        self.angles = self.init_angles()

        # Data sets and data loaders
        self.datasets = self.init_datasets()

        # data_loaders
        self.data_loaders = dict().fromkeys(['train', 'val', 'test'])

        # Model
        try:
            self.model = gn.GaugeNet.load(directory=self.directory)
            typer.secho(f'Model loaded from {self.directory}', fg='green')
            train = False or force_train

        except FileNotFoundError:
            train = True

        if settings.DEV != 'True' and train:
            train = typer.confirm(
                f'Start training for this gauge (either transfer learning or learning from scratch)?',
                default=True,
                abort=True)

        if train:
            self.model = gn.GaugeNet(directory=self.directory)
            self.train(transfer_learning=False)

    def init_angles(self):
        min_angle = float(self.calibration['needle']['min_angle'])
        max_angle = float(self.calibration['needle']['max_angle'])
        train_angles = np.linspace(min_angle, max_angle, settings.IMAGE_TRAIN_SET_SIZE)
        val_angles = np.random.uniform(min_angle, max_angle, settings.IMAGE_VAL_SET_SIZE)
        test_angles = np.random.uniform(min_angle, max_angle, settings.IMAGE_TEST_SET_SIZE)
        angles = {'train': train_angles, 'val': val_angles, 'test': test_angles}
        return angles

    def init_datasets(self,
                      sets: list = ('train', 'val', 'test')):
        datasets = {}
        for set_type in sets:
            datasets[set_type] = img_dataset.AnalogDataSet(set_type=set_type,
                                                           base_image=self.base_image,
                                                           needle_image=self.needle_image,
                                                           angles=self.angles[set_type],
                                                           calibration=self.calibration)
        return datasets

    def create_train_val_set(self):
        for set_type in ['train', 'val', 'test']:
            self.datasets[set_type].create_dataset()
        return None

    def init_data_loaders(self,
                          sets: list = ('train', 'val', 'test')):
        for set_type in sets:
            self.data_loaders[set_type] = DataLoader(self.datasets[set_type],
                                                     batch_size=settings.BATCH_SIZE,
                                                     shuffle=False)

    def train(self,
              transfer_learning: bool = False):
        self.init_data_loaders(sets=['train', 'val', 'test'])
        self.model.to(settings.DEVICE)
        typer.secho(f'Training model on {settings.DEVICE}, '
                    f'Camera: {self.calibration["camera_id"]} '
                    f'Gauge index: {self.calibration["index"]} ', fg=typer.colors.BRIGHT_MAGENTA)
        self.model.train_sequence(train_loader=self.data_loaders['train'],
                                  val_loader=self.data_loaders['val'],
                                  test_loader=self.data_loaders['test'],
                                  transfer_learning=transfer_learning)
        return None

    def visual_test(self,
                    model: gn.GaugeNet = None):
        """
        Test the model's performance.
        """
        typer.secho(f'Testing model on {settings.DEVICE}', fg=typer.colors.CYAN)
        if model is None:
            model = self.model
        test_path = os.path.join(self.directory, 'test')
        test_images = [os.path.join(test_path, x) for x in os.listdir(test_path)]
        size = settings.TEST_REPORT_IMAGE_TILE
        fig_shape = [size] * 2
        fig_size = [settings.REPORT_PLT_SIZE] * 2
        fig = plt.figure(figsize=fig_size)
        for i, image in enumerate(start=1, iterable=test_images):
            if i > size * size:
                break
            pred_value = self.get_reading(frame=image,
                                          model=model,
                                          restore_edit_steps=False,
                                          prints=False)
            fig.add_subplot(fig_shape[0], fig_shape[1], i)
            figure = cv2.imread(image)
            plt.imshow(figure)
            plt.axis('off')
            pred_value = round(pred_value, 2)
            plt.title(f'Predicted Value: \n {pred_value} {self.calibration["units"]}')
        fig.suptitle(f'Camera: {self.calibration["camera_id"]} Gauge: {self.calibration["index"]} Test Results',
                     fontsize=24)
        # enlarge spacing between subplots
        fig.subplots_adjust(hspace=0.5)
        fig.savefig(os.path.join(self.directory, settings.REPORT_PLT_NAME))
        typer.echo(f'Test results saved to {self.directory}')

    def get_reading(self,
                    frame: str or np.ndarray,
                    model: gn.GaugeNet = None,
                    restore_edit_steps: bool = True,
                    prints: bool = True):
        """
        Get the reading of the gauge.
        :param model:
        :param frame:
        :param restore_edit_steps: Perform the edit steps as saved in the XML file
        :param prints: Print the results
        :return:
        """
        model = model if model else self.model
        crop_coords = None
        perspective_pts = None
        perspective_changed = False
        if isinstance(frame, str):
            path = (Path(settings.FRAMES_PATH) / frame).as_posix()
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise FileNotFoundError(f'Image not found: {path}')
        if restore_edit_steps:
            crop_coords = self.calibration['crop']
            crop_coords = [int(x) for x in crop_coords]
            perspective_pts = self.calibration['perspective']
            if isinstance(perspective_pts[0], str):
                perspective_pts = [pts.strip('[').strip(']').split(',') for pts in perspective_pts]
            perspective_pts = [tuple(int(x) for x in pts) for pts in perspective_pts]
            perspective_changed = self.calibration['perspective_changed']
            perspective_changed = True if perspective_changed == 'True' else False
        image = ie.frame_to_read_image(frame=frame,
                                       crop_coords=crop_coords,
                                       perspective_pts=perspective_pts,
                                       perspective_changed=perspective_changed)
        rad = model(image)
        reading = self.get_value(rad=rad)
        if prints:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            typer.echo('Time: {} | Gauge: {} | Camera: {} | Reading: {:.2f} {}'.format(time,
                                                                                       self.calibration['index'],
                                                                                       self.calibration['camera_id'],
                                                                                       reading,
                                                                                       self.calibration['units']))
        return reading

    def get_value(self,
                  rad: torch.Tensor):
        """
        Converts angle to value
        """
        min_angle = float(self.calibration['needle']['min_angle'])
        angle = np.rad2deg(rad.item())
        if angle > 0:
            min_rel_angle = min_angle - angle
        else:
            min_rel_angle = min_angle + abs(angle)
        value_step = float(self.calibration['step_value'])
        value = min_rel_angle * value_step
        min_val = float(self.calibration['min_value'])
        return min_val + value

    @classmethod
    def calibrate(cls,
                  calibration_image: str = None,
                  index: int = 1,
                  camera_id: int = 1):
        """
        Call the Calibration APP for the gauge.
        :param calibration_image:
        :param index: Index of the gauge.
        :param camera_id: Camera ID of the gauge.
        :return: None
        """
        directory_path, index = env.set_gauge_directory(index, camera_id)
        calibrator_app = calibrator.AnalogCalibrator()
        calibration = calibrator_app.run(index=index,
                                         camera_id=camera_id,
                                         frame_name=calibration_image,
                                         directory=directory_path)
        return calibration
