import cv2
import os
import src.utils.envconfig as env


class Camera:
    def __init__(self,
                 camera_id: str):
        self.camera_id = camera_id
        self.path = os.path.join(env.DEFAULT_PATH, camera_id)
        self._init_directory()

        # Gauges in frames
        self.gauges = {}
        self.gauges_count = 0
        self.gauges_max = env.GAUGES_MAX

    def _init_directory(self):  # TODO: add overwrite option
        if not os.path.exists(self.path):
            os.mkdir(self.path)

