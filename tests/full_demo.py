import src.gauges.gauge as g
import src.model.gauge_net as gn


calibration = g.AnalogGauge.calibrate()
analog_gauge = g.AnalogGauge(calibration)
model = gn.GaugeNet()
analog_gauge.train(model=model)
analog_gauge.get_reading(frame='Speed.jpg', model=model)

