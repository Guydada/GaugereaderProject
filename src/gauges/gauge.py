import os
import typer
import src.utils.convert_xml as xmlr
import src.utils.envconfig as env
import src.calibrator.app as calibrator


class Gauge:
    def __init__(self,
                 timestamp: str,
                 camera_id: str,
                 index: str,
                 description: str,
                 gauge_type: str = 'analog',
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
        if ui_calibration:  # TODO: prompt for calibration if given
            self.ui_calibrate()
        elif calibration_file is not None:  # TODO: Parse XML file to get the same dictionary
            self.calibrate_from_xml()

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
        self.calibration = xmlr.xml_to_dict(self.xml_file)
        return None

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
