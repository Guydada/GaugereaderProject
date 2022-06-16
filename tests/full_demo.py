import src.gauges.gauge as g
import src.model.gauge_net as gn


calibration = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge(calibration)
model = gn.GaugeNet()
analog_gauge.train(model=model)
model.save()
analog_gauge.get_reading(frame='IMG_1681.jpg', model=model)


