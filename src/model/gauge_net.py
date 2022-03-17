import os
import datetime
import torch
import typer
import torch.nn as nn
import pandas as pd
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
        self.train_report = {'epoch': [], 'loss': []}

    def train_sequence(self,
                       train_loader,
                       directory,
                       epochs: int = 100):
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
            self.train_report['loss'].append(avg_loss)
        path = os.path.join(directory, 'train_report.csv')
        train_report_df = pd.DataFrame(self.train_report)
        train_report_df.to_csv(path, index=False)
        # self.save() # TODO: save model to disk if needed

    def test_sequence(self,
                      test_loader,
                      directory):
        test_report = pd.DataFrame()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = self.criterion(output, target.reshape([env.BATCH_SIZE, 1]).float())
                test_report['angle'] = target.cpu().numpy()
                test_report['prediction'] = output.squeeze().cpu().numpy()
        path = os.path.join(directory, 'test_report.csv')
        test_report.to_csv(path, index=False)
        typer.echo('Test loss: %.3f' % loss.item())

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

