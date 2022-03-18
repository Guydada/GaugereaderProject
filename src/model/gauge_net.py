import os
import datetime
import torch
import typer
import torch.nn as nn
import pandas as pd
import numpy as np
import src.utils.envconfig as env


input_size = env.TRAIN_IMAGE_SIZE * env.TRAIN_IMAGE_SIZE * 3
hidden_layer1 = 1024


class GaugeNet(nn.Module):
    def __init__(self):
        super(GaugeNet, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=16,
                                              kernel_size=(5, 5),
                                              stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(in_channels=16,
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
                                    nn.Flatten(),
                                    nn.Linear(in_features=1024,
                                              out_features=256),
                                    nn.ReLU(),
                                    nn.Linear(in_features=256,
                                              out_features=128),
                                    nn.ReLU(),
                                    nn.Linear(in_features=128,
                                              out_features=1))
        for i in [-1, -2]:
            if self.layers[i]._get_name() is not 'ReLU':
                nn.init.xavier_uniform_(self.layers[i].weight)

        self.device = env.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train_report = {'epoch': [], 'train_loss': []}

    def train_sequence(self,
                       train_loader,
                       directory,
                       epochs: int = env.EPOCHS):
        for epoch in range(epochs):
            running_loss = 0.0
            count = 1
            for i, (images, angles) in enumerate(train_loader):
                images = images.to(self.device)
                angles = angles.to(self.device)
                self.optimizer.zero_grad()
                output = self(images)
                loss = self.criterion(output, angles.reshape([env.BATCH_SIZE, 1]).float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, \t %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                count += 1
            avg_loss = running_loss / count
            print('Finished epoch %d' % (epoch + 1), '\n'
                  'Training loss: %.3f' % avg_loss)
            self.train_report['epoch'].append(epoch + 1)
            self.train_report['train_loss'].append(avg_loss)
        path = os.path.join(directory, 'train_report.csv')
        train_report_df = pd.DataFrame(self.train_report)
        train_report_df.to_csv(path, index=False)
        path = os.path.join(directory, 'test_report.csv')
        # self.save() # TODO: save model to disk if needed

    def test_sequence(self,
                      directory,
                      test_loader):
        test_report = pd.DataFrame(columns=['real_angle', 'predicted_angle'])
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = self.criterion(output, target.reshape([env.BATCH_SIZE, 1]).float())
                df = np.stack([target.cpu().numpy(), output.squeeze().cpu().numpy()], axis=1)
                test_report = test_report.append(pd.DataFrame(df, columns=['real_angle', 'predicted_angle']))
        typer.echo('Test loss: %.3f' % loss.detach())
        test_report_df = pd.DataFrame(test_report)
        path = os.path.join(directory, 'test_report.csv')
        test_report_df.to_csv(path, index=False)
        return test_report_df

    def save(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(env.MODELS_PATH, f'gauge_net_{timestamp}.pt')
        torch.save(self, path)

    def forward(self, x):
        x = self.layers(x)
        return x.type(torch.float)

    @classmethod
    def load(cls, path):
        return torch.load(path)

