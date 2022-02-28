import os
import src.utils.convert_xml as xmlr
import typer
import torch
from pathlib import Path

# Roots and directories
PROJECT_ROOT = Path(__file__).parent.parent.parent            # Root of the project
SAVED_MODELS_DIR = PROJECT_ROOT.joinpath("models")            # Directory for saved models
DEFAULT_PATH = PROJECT_ROOT.joinpath("data")                  # Default path to the data directory
CALIBRATION_PATH = DEFAULT_PATH.joinpath("calibration_data")  # Path to the calibration_data directory
CAMERAS_PATH = DEFAULT_PATH.joinpath("cameras")               # Path to the cameras directory
FRAMES_PATH = DEFAULT_PATH.joinpath("frames")                 # Path to the frames directory
CAMERA_LIST_FILE = CAMERAS_PATH.joinpath("camera_list.xml")   # Path to the camera list file
SAVE_EXT = '.pkl'                                             # Extension for the saved files
DEV_CAM = 'camera_001'                                        # Name of the development camera
DEV_GAUGE = 'gauge_001'                                       # Name of the development gauge
DEV_CALIBRATION_PHOTO = 'Speed.jpg'                           # Name of the development calibration_data photo
GAUGE_CALIBRATION_FILE_XML = 'gauge_params.xml'               # Name of the gauge calibration_data file
TRAIN_IMAGE_NAME = 'train_image.jpg'                          # Name of the training image
NEEDLE_IMAGE_NAME = 'needle_image.jpg'                        # Name of the needle image
TRAIN_SET_DIR_NAME = 'train_set'                              # Name of the training set directory
VALIDATION_SET_DIR_NAME = 'validation_set'                    # Name of the validation set directory
dir_list = [DEFAULT_PATH,                                     # List of directories to create
            CALIBRATION_PATH,
            CAMERAS_PATH,
            FRAMES_PATH]

DEV_CALIBRATION_PHOTO_PATH = CALIBRATION_PATH.joinpath(DEV_CALIBRATION_PHOTO)
DEV_CALIBRATION_FILE_XML = CALIBRATION_PATH.joinpath(GAUGE_CALIBRATION_FILE_XML)

# UI parameters
WINDOW_SIZE = (500, 500)
EDIT_IMAGE_SIZE = (400, 400)
TRAIN_IMAGE_SIZE = 256

# MODEL parameters
BATCH_SIZE = 64
EPOCHS = 10
NUM_WORKERS = 1
IMAGE_TRAIN_SET_SIZE = BATCH_SIZE * 2
IMAGE_TEST_SET_SIZE = BATCH_SIZE

# Circle detection parameters

# Gauge Types
GAUGE_TYPES = ['analog', 'digital']

# Torch parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Functions
def dir_file_from_camera_gauge(camera_id: str,
                               gauge_id: str) -> (str, str):
    """
    Return the directory for a camera and gauge
    :param camera_id:
    :param gauge_id:
    :return:
    """
    directory = CAMERAS_PATH.joinpath(camera_id).joinpath(gauge_id)
    file = directory.joinpath(GAUGE_CALIBRATION_FILE_XML)
    return directory.as_posix(), file.as_posix()


def check_dirs():
    """
    Check if the directories exist, and create them if they don't
    :return:
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)


def str_to_calib(calibration_image: str) -> str:
    """
    Convert a calibration_data image name to a full os POSIX path
    :param calibration_image:
    :return:
    """
    path = CALIBRATION_PATH.joinpath(calibration_image).as_posix()
    return path


def create_gauge_dict(camera_id: str,
                      gauge_id: str,
                      calibration_image: str):
    """
    Create a gauge dictionary.
    """
    gauge = dict(gauge_id=gauge_id,
                 calibration_image=calibration_image,
                 calibration_file=dir_file_from_camera_gauge(camera_id, gauge_id))
    return gauge


def create_camera_dict(camera_id: str,
                       ship_location: str,
                       gauges: dict = None):
    """
    Create a camera dictionary.
    """
    gauges = gauges if gauges is not None else {}
    camera = dict(camera_id=camera_id,
                  ship_location=ship_location,
                  gauges=gauges)
    return camera


def add_to_camera_list(camera_id: str,
                       ship_location: str,
                       gauges: dict = None):
    """
    Add a camera to the camera list file.
    """
    path = CAMERA_LIST_FILE.as_posix()
    camera_list = xmlr.xml_to_dict(path)
    camera_list[camera_id] = create_camera_dict(camera_id,
                                                ship_location,
                                                gauges)
    xmlr.dict_append_to_xml(camera_list, path)
    typer.echo(f"Added camera {camera_id} to camera list file.")
    return


def create_camera_list(dev: bool = True):
    """
    Create a camera list file.
    """
    path = CAMERA_LIST_FILE.as_posix()
    cameras = {}
    if dev:
        cameras[DEV_CAM] = create_camera_dict(DEV_CAM,
                                              'Dev',
                                              {DEV_GAUGE: create_gauge_dict(DEV_CAM,
                                                                            DEV_GAUGE,
                                                                            DEV_CALIBRATION_PHOTO)})
    if os.path.exists(path):
        xmlr.dict_append_to_xml(cameras, path)
        typer.secho(f"Appended camera list to {path}", fg=typer.colors.GREEN)

    else:
        xmlr.dict_to_xml(cameras, path)
        typer.secho(f"Created camera list file {path}", fg=typer.colors.GREEN)
    return


def set_env():
    """
    Initial environment creation, including directories and variables
    :return:
    """
    check_dirs()
    create_camera_list()
    return None


