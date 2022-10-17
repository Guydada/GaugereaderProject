import sys
from pathlib import Path

FILE = Path(__file__).parent.parent.resolve()
if FILE not in sys.path:
    sys.path.append(str(FILE))

import src.gauges.gauge as g

# calibration = g.AnalogGauge.calibrate()
# analog_gauge = g.AnalogGauge(calibration)
analog_gauge = g.AnalogGauge('camera_1_analog_gauge_1.xml')
analog_gauge.initialize(force_train=True)
analog_gauge.visual_test()
analog_gauge.get_reading('demo.jpg')
