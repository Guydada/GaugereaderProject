import os
import typer
import torch
from pathlib import Path

# Roots and directories
PROJECT_ROOT = Path(__file__).parent.parent.parent               # Root of the project
DEFAULT_PATH = PROJECT_ROOT.joinpath("data")                     # Default path to the data directory
GAUGES_PATH = DEFAULT_PATH.joinpath("gauges")                    # Path to the calibration_data directory
MODELS_PATH = DEFAULT_PATH.joinpath("models")                    # Path to the models' directory
FRAMES_PATH = DEFAULT_PATH.joinpath("frames")                    # Path to the frames directory
SAVE_EXT = '.pkl'                                                # Extension for the saved files
DEV_CAM = 'camera_001'                                           # Name of the development camera
DEV_GAUGE = 'gauge_001'                                          # Name of the development gauge
DEV_CALIBRATION_PHOTO = 'Speed.jpg'                              # Name of the development calibration_data photo
GAUGE_CALIBRATION_FILE_XML = 'gauge_params.xml'                  # Name of the gauge calibration_data file
TRAIN_IMAGE_NAME = 'train_image.jpg'                             # Name of the training image
TEST_IMAGE_NAME = 'test_image.jpg'                               # Name of the training image
NEEDLE_IMAGE_NAME = 'needle_image.jpg'                           # Name of the needle image
TRAIN_SET_DIR_NAME = 'train_set'                                 # Name of the training set directory
XML_FILE_NAME = 'gauge_params.xml'                               # Name of the gauge calibration_data file
VALIDATION_SET_DIR_NAME = 'validation_set'                       # Name of the validation set directory
dir_list = [DEFAULT_PATH,  # List of directories to create
            GAUGES_PATH,
            MODELS_PATH,
            FRAMES_PATH]

DEV_CALIBRATION_PHOTO_PATH = GAUGES_PATH.joinpath(DEV_CALIBRATION_PHOTO)
DEV_CALIBRATION_FILE_XML = GAUGES_PATH.joinpath(GAUGE_CALIBRATION_FILE_XML)

# UI parameters
WINDOW_SIZE = (500, 500)
EDIT_IMAGE_SIZE = (400, 400)
TRAIN_IMAGE_SIZE = 64

# MODEL parameters
BATCH_SIZE = 64
EPOCHS = 100
NUM_WORKERS = 1
IMAGE_TRAIN_SET_SIZE = BATCH_SIZE * 2
IMAGE_TEST_SET_SIZE = BATCH_SIZE

# Circle detection parameters

# Gauge Types
GAUGE_TYPES = ['analog', 'digital']

# Torch parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_dirs():
    """
    Check if the directories exist, and create them if they don't
    :return:
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)


def set_env():
    """
    Initial environment creation, including directories and variables
    :return:
    """
    check_dirs()


def set_gauge_directory(index: int,
                        camera_id: int):
    """
    Set the directory for the gauge
    :param index: Index of the gauge
    :param camera_id: Camera ID of the gauge
    :return:
    """
    gauge_dir = GAUGES_PATH.joinpath(f'camera_{camera_id}')
    if not gauge_dir.exists():
        gauge_dir.mkdir()
    gauge_dir = gauge_dir.joinpath(f'gauge_{index}')
    if not gauge_dir.exists():
        gauge_dir.mkdir()
    else:
        msg = f'Directory {gauge_dir} already exists. Do you want to overwrite it? if not, index will increment'
        confirm = typer.confirm(msg, default=True)
        if not confirm:
            gauge_dir = set_gauge_directory(index + 1, camera_id)
    return gauge_dir
