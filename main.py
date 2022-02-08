import typer
import datetime

import src.reader.input_output as io
import src.utils.envconfig as env
import src.utils.convert_xml as xml


def add_camera(camera_id: str):
    """
    Add an empty (no gauges) camera to the dashboard.
    :param camera_id: Camera ID - unique identifier for each camera in a ship
    :return: None
    """
    new_camera = env.create_camera_dict(camera_id)
    xml.dict_append_to_xml(new_camera, env.CAMERA_LIST_FILE)
    typer.echo(f"Added camera {camera_id} to dashboard.")
    return None


def add_gauge(timestamp: str,
              camera_id: str,
              index: str,
              description: str,
              gauge_type: str,
              ui_calibration: bool = True,
              calibration_image: str = None,
              calibration_file: str = None):
    """
    Add a gauge to the dashboard.
    :param timestamp: Time of the frame
    :param camera_id: Camera ID - unique identifier for each camera in a ship
    :param index: Gauge index - unique inside each camera
    :param description: Description of the gauge - free text
    :param gauge_type: Gauge type (e.g 'analog', 'digital')
    :param ui_calibration: If True, the gauge will be calibrated in the Calibrator app
    :param calibration_image: Name of the calibration image. loaded from a default path in src.utils.envconfig
    :param calibration_file: If given, the calibration file will be used - XML file
    :return: g.Gauge object
    """
    pass


def read_frame(camera_id: str,
               timestamp: str,
               frame_id: str):
    """
    Read a frame, given the camera ID, timestamp and frame, update the reading history for each gauge
    :param camera_id: Camera ID - unique identifier for each camera in a ship
    :param timestamp: Time of the frame
    :param frame_id: Frame ID - For identifying frames with the same timestamp
    :return: A io.FrameOutput object containing the reading Dataclass objects of each gauge in the frame
    """
    pass
