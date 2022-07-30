import torch
from pathlib import Path

from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

# Roots and directories
settings.PROJECT_ROOT = Path(__file__).parent  # Root of the project
settings.DEFAULT_PATH = settings.PROJECT_ROOT.joinpath("data")  # Default path to the data directory
settings.GAUGES_PATH = settings.DEFAULT_PATH.joinpath("gauges")  # Path to the calibration_data directory
settings.XML_FILES_PATH = settings.DEFAULT_PATH.joinpath("xml_files")  # Path to the xml_files directory
settings.MODELS_PATH = settings.DEFAULT_PATH.joinpath("models")  # Path to the models' directory
settings.FRAMES_PATH = settings.DEFAULT_PATH.joinpath("frames")  # Path to the frames directory
settings.DEV_CALIBRATION_FILE_XML = settings.GAUGES_PATH.joinpath(settings.GAUGE_CALIBRATION_FILE_XML)
settings.dir_list = [settings.DEFAULT_PATH,  # List of directories to create
                     settings.GAUGES_PATH,
                     settings.MODELS_PATH,
                     settings.XML_FILES_PATH,
                     settings.FRAMES_PATH]

# Torch parameters
settings.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image Parameters
settings.TRAIN_IMAGE_SHAPE = [settings.TRAIN_IMAGE_SIZE] * 2  # Shape of the images

# Train Settings
settings.IMAGE_TRAIN_SET_SIZE = settings.BATCH_SIZE * 3
settings.IMAGE_VAL_SET_SIZE = settings.BATCH_SIZE
settings.IMAGE_TEST_SET_SIZE = settings.BATCH_SIZE
