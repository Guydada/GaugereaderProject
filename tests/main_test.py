import datetime
import torch

import src.model.gauge_net as gn
import src.gauges.gauge as g
import src.utils.envconfig as env


analog_gauge = g.AnalogGauge()


# model = gn.GaugeNet()
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# #
# #
# analog_gauge.train(model=model,
#                    criterion=criterion,
#                    optimizer=optimizer,
#                    epochs=1)

