from dataclasses import dataclass


@dataclass
class Calibration:
    """
    Calibration class.
    """
    read_method: str = 'unknown'
    step: float = 0.0
    units: str = 'unknown'

    def as_dict(self):
        """
        Return a dictionary representation of the calibration_data object.
        """
        return self.__dict__

    def __getitem__(self, key):
        """
        Return the value of the calibration_data object's key.
        return self.__dict__[key]
        :param key:
        :return:
        """
        return self.__dict__[key]

    def __repr__(self):
        """
        Return a string representation of the calibration_data object.
        """
        return str(self.__dict__)

    def __str__(self):
        """
        Return a string representation of the calibration_data object.
        """
        return str(self.__dict__)


@dataclass
class AnalogCalibration(Calibration):
    """
    Class to hold calibration_data data for an analog sensor.
    """
    min_value: int = 0
    max_value: int = 30
    min_radius: float = 0.0
    max_radius: float = 0.0
    min_distance: float = 0.0
    step: float = 0.0


@dataclass
class DigitalCalibration(Calibration):
    """
    Class to hold calibration_data data for a digital/text sensor.
    """
    pass

