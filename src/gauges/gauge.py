import os
import numpy as np
import cv2

from torch.utils.data import DataLoader

import src.model.dataset_class as img_dataset
import src.calibrator.app as calibrator
import src.utils.convert_xml as xmlr
import src.utils.envconfig as env
import src.utils.image_editing as ie


class Gauge:
    def __init__(self,
                 calibration: dict or str = None):
        """
        Initialize the gauge.
        :param calibration: Calibration dictionary or path to the calibration xml file.
        """
        if isinstance(calibration, dict):
            self.calibration = calibration
        elif isinstance(calibration, str):
            self.calibration = xmlr.xml_to_dict(calibration)
            self.calibration = self.calibration['gauge']
        # Inner variables
        self.directory = self.calibration['directory']

    def create_train_test_set(self,
                              train_size: float = 0.8,
                              test_size: float = 0.2,
                              random_state: int = 42):
        pass

    def get_reading(self,
                    frame):
        """
        Get the reading of the gauge.
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
        self.train_set_images_dir = os.path.join(self.directory, 'train')
        self.test_set_images_dir = os.path.join(self.directory, 'test')
        self.train_image_path = os.path.join(self.directory, env.TRAIN_IMAGE_NAME)
        self.needle_image_path = os.path.join(self.directory, env.NEEDLE_IMAGE_NAME)

        # Train/test base images
        self.base_image = cv2.imread(self.train_image_path)
        if self.base_image is None:
            raise FileNotFoundError(f'Train image "{self.base_image}" not found')
        self.needle_image = cv2.imread(self.needle_image_path)
        if self.needle_image is None:
            raise FileNotFoundError(f'Needle image "{self.needle_image}" not found')

        # Angles
        self.angles = self.init_angles()

        # Data sets and data loaders
        self.datasets = self.init_datasets()

        # data_loaders
        self.data_loaders = dict().fromkeys(['train', 'test'])

    def init_angles(self):
        min_angle = float(self.calibration['needle']['min_angle'])
        max_angle = float(self.calibration['needle']['max_angle'])
        train_angles = np.linspace(min_angle, max_angle, env.IMAGE_TRAIN_SET_SIZE)
        test_angles = np.random.uniform(min_angle, max_angle, env.IMAGE_TEST_SET_SIZE)
        angles = {'train': train_angles, 'test': test_angles}
        return angles

    def init_datasets(self):
        datasets = {}
        for set_type in ['train', 'test']:
            datasets[set_type] = img_dataset.AnalogDataSet(set_type=set_type,
                                                           base_image=self.base_image,
                                                           needle_image=self.needle_image,
                                                           angles=self.angles[set_type],
                                                           calibration=self.calibration)
        return datasets

    def create_train_test_set(self,
                              train_size: float = 0.8,
                              test_size: float = 0.2,
                              random_state: int = 42):
        for set_type in ['train', 'test']:
            self.datasets[set_type].create_dataset()
        return None

    def train(self,
              model):
        for set_type in ['train', 'test']:
            self.data_loaders[set_type] = DataLoader(self.datasets[set_type],
                                                     batch_size=env.BATCH_SIZE,
                                                     shuffle=False)
        model.to(env.DEVICE)
        model.train_sequence(train_loader=self.data_loaders['train'],
                             directory=self.directory)
        model.test_sequence(test_loader=self.data_loaders['test'],
                            directory=self.directory)
        return None

    def get_reading(self,
                    model,
                    frame: np.array = None):
        crop_coords = self.calibration['crop']
        perspective_pts = self.calibration['perspective']
        image = ie.frame_to_read_image(frame=frame,
                                       crop_coords=crop_coords,
                                       perspective_pts=perspective_pts)
        # get reading
        reading = model(image)
        return reading

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
