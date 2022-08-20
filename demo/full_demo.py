import src.gauges.gauge as g

# calibration = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge('camera_1_analog_gauge_5.xml')
analog_gauge.start()
analog_gauge.visual_test()
