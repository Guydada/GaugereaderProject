import cv2
from dataclasses import dataclass


@dataclass
class GaugeOutput:
    time_stamp: str
    camera_id: str
    gauge_id: str
    value: float
    units: str

    def __str__(self):
        return f"{self.time_stamp} " \
               f"{self.camera_id} " \
               f"{self.gauge_id} " \
               f"{self.value} {self.units}"