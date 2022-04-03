import src.gauges.gauge as g

calibration = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge(calibration)

