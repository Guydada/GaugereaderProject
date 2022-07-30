import src.gauges.gauge as g

calibration_1 = 'camera_1_analog_gauge_1.xml'
calibration_2 = 'camera_1_analog_gauge_6.xml'

for cal in [calibration_1, calibration_2]:
    analog_gauge = g.AnalogGauge(cal)
    analog_gauge.visual_test()