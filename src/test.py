import datetime

import src.gauges.gauge as g
import src.utils.envconfig as env


dev_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


analog_gauge = g.AnalogGauge(timestamp=dev_timestamp,
                             camera_id=env.DEV_CAM,
                             index=env.DEV_GAUGE,
                             description="Test gauge",
                             ui_calibration=True,
                             # calibration_image=env.DEV_CALIBRATION_PHOTO)
                             calibration_image=env.DEV_CALIBRATION_PHOTO,
                             calibration_file=env.DEV_CALIBRATION_FILE_XML)


analog_gauge.create_train_test_set()
