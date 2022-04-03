import src.model.gauge_net as gn
import src.gauges.gauge as g


xml_name = 'camera_1_analog_gauge_2.xml'
analog_gauge = g.AnalogGauge(xml_name)
model = gn.GaugeNet()
analog_gauge.train(model=model)
model.save()
