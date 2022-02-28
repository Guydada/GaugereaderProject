
import torch
import torch.nn as nn

import src.utils.envconfig as env


input_size = env.TRAIN_IMAGE_SIZE * env.TRAIN_IMAGE_SIZE * 3
hidden_layer1 = 1024


class GaugeNet(nn.Module):
    def __init__(self):
        super(GaugeNet, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=32,
                                              kernel_size=(5, 5),
                                              stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=64,
                                              kernel_size=(5, 5),
                                              stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=(5, 5),
                                              stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Flatten(),
                                    nn.Linear(in_features=100352,
                                              out_features=512),
                                    nn.Linear(in_features=512,
                                              out_features=1)
                                    )
        for i in [-1, -2]:
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, x):
        x = self.layers(x)
        return x.type(torch.float)
