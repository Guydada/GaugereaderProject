import os
import typer
import torch
from pathlib import Path

# Roots and directories
PROJECT_ROOT = Path(__file__).parent.parent.parent               # Root of the project
DEFAULT_PATH = PROJECT_ROOT.joinpath("data")                     # Default path to the data directory
GAUGES_PATH = DEFAULT_PATH.joinpath("gauges")                    # Path to the calibration_data directory
XML_FILES_PATH = DEFAULT_PATH.joinpath("xml_files")              # Path to the xml_files directory
MODELS_PATH = DEFAULT_PATH.joinpath("models")                    # Path to the models' directory
FRAMES_PATH = DEFAULT_PATH.joinpath("frames")                    # Path to the frames directory
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
            XML_FILES_PATH,
            FRAMES_PATH]

DEV_CALIBRATION_FILE_XML = GAUGES_PATH.joinpath(GAUGE_CALIBRATION_FILE_XML)

# UI parameters
WINDOW_SIZE = (1500, 1500)
EDIT_IMAGE_SIZE = (400, 400)
TRAIN_IMAGE_SIZE = 64

# MODEL parameters
BATCH_SIZE = 64
EPOCHS = 100
NUM_WORKERS = 1
IMAGE_TRAIN_SET_SIZE = BATCH_SIZE * 3
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
    :return: Path to the gauge directory, index of the gauge
    """
    set_env()
    gauge_dir = GAUGES_PATH.joinpath(f'camera_{camera_id}', f'gauge_{index}')
    if os.path.exists(gauge_dir):
        index += 1
        typer.secho(f'Gauge {index} already exists, creating gauge {index + 1}', fg='yellow')
        return set_gauge_directory(index, camera_id)
    os.makedirs(gauge_dir)
    return gauge_dir, index
