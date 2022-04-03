import src.model.gauge_net as gn
import src.gauges.gauge as g


for i in range(1, 3):
    if i == 1:
        model = gn.GaugeNet()
    else:
        model = gn.GaugeNet.load()
    xml_name = 'camera_1_analog_gauge_{}.xml'.format(i)
    analog_gauge = g.AnalogGauge(xml_name)
    analog_gauge.train(model=model)
    model.save()
