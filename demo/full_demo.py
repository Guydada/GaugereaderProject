import src.gauges.gauge as g

# analog_gauge = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge('camera_1_analog_gauge_1.xml')
analog_gauge.initialize(force_train=True)
analog_gauge.visual_test()
