import datetime
import torch

import src.model.gauge_net as gn
import src.gauges.gauge as g
import src.utils.envconfig as env


xml_name = 'gauge_config.xml'
analog_gauge = g.AnalogGauge(xml_file)
analog_gauge.create_train_test_set()
model = gn.GaugeNet()
analog_gauge.train(model=model)
