import typer
import datetime

import src.gauges.gauge as g
import src.utils.envconfig as env
import src.utils.convert_xml as xml

env.set_env()


def add_camera(camera_id: str):
    """
    Add an empty (no gauges) camera to the dashboard.
    """
    new_camera = env.create_camera_dict(camera_id)
    xml.dict_append_to_xml(new_camera, env.CAMERA_LIST_FILE)
    typer.echo(f"Added camera {camera_id} to dashboard.")
    return camera_id


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
    """
    gauge = g.Gauge(timestamp,
                    camera_id,
                    index,
                    description,
                    gauge_type,
                    ui_calibration,
                    calibration_image,
                    calibration_file)

    gauge_dict = gauge.as_dict()
    camera_list = xml.xml_to_dict(env.CAMERA_LIST_FILE)
    camera_list[camera_id]['gauges'].append(gauge_dict)
    xml.dict_append_to_xml(camera_list, env.CAMERA_LIST_FILE)
    typer.echo(f"Added gauge {gauge.index} to camera {camera_id}.")
    return gauge


def read_frame(camera_id: str,
               timestamp: str,
               frame_id: str):
    """
    Read a all gauges in a frame from a camera.  # TODO check if I need to separate the reading of same frame gauges
    """
    pass


# create example timestamp
dev_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

add_gauge(timestamp=dev_timestamp,
          camera_id=env.DEV_CAM,
          index=env.DEV_GAUGE,
          description="Test gauge",
          gauge_type="analog",
          ui_calibration=True,
          calibration_image=env.DEV_CALIBRATION_PHOTO)


