import os
import typer

from config import settings


def check_dirs():
    """
    Check if the directories exist, and create them if they don't
    :return: None
    """
    lst = list(settings.dir_list)
    for directory in lst:
        if not os.path.exists(directory):
            os.mkdir(directory)


def set_env():
    """
    Initial environment creation, including directories and variables
    :return: None
    """
    check_dirs()


def set_gauge_directory(index: int,
                        camera_id: int,
                        changed: bool = False):
    """
    Set the directory for the gauge
    :param changed:
    :param index: Index of the gauge
    :param camera_id: Camera ID of the gauge
    :return: Path to the gauge directory, index of the gauge
    """
    set_env()
    gauge_dir = settings.GAUGES_PATH.joinpath(f'camera_{camera_id}', f'gauge_{index}')
    if os.path.exists(gauge_dir):
        index += 1
        changed = True
        return set_gauge_directory(index, camera_id, changed)
    if changed:
        typer.secho(f'Gauge already exists, index re-assigned automatically to {index}', fg='yellow')
    typer.secho(f'Gauge directory set to {gauge_dir}', fg='green')
    os.makedirs(gauge_dir)
    return gauge_dir, index


def get_directory(index: int,
                  camera_id: int):
    """
    Set the directory for the gauge
    :param index: Index of the gauge
    :param camera_id: Camera ID of the gauge
    :return: Path to the gauge directory, index of the gauge
    """
    index, camera_id = int(index), int(camera_id)
    gauge_dir = settings.GAUGES_PATH.joinpath(f'camera_{camera_id}', f'gauge_{index}')
    return gauge_dir
