import datetime
import torch

import src.model.gauge_net as gn
import src.gauges.gauge as g
import src.utils.envconfig as env


calibration = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge(calibration)

