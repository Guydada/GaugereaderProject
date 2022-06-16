import src.model.gauge_net as gn
import src.gauges.gauge as g

model = gn.GaugeNet.load()

xml_name = 'camera_1_analog_gauge_2.xml'
analog_gauge = g.AnalogGauge(xml_name)
analog_gauge.get_reading(frame='IMG_1681.jpg', model=model)
