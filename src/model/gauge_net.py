import os
import datetime
import torch
import typer
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.utils.envconfig as env

torch.manual_seed(env.TORCH_SEED)


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
                                              out_features=512),
                                    nn.Linear(in_features=512,
                                              out_features=256),
                                    nn.ReLU(),
                                    nn.Linear(in_features=256,
                                              out_features=128),
                                    nn.ReLU(),
                                    nn.Linear(in_features=128,
                                              out_features=1))
        for i in [-1, -2]:
            if self.layers[i]._get_name() not in ['ReLU', 'Dropout']:
                nn.init.xavier_uniform_(self.layers[i].weight)
        self.device = env.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=env.LEARNING_RATE)
        self.train_report = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    def train_one_epoch(self,
                        train_loader):
        running_loss = 0.0
        for i, (images, angles) in enumerate(train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            self.optimizer.zero_grad()
            output = self(images)
            loss = self.criterion(output, angles.reshape([env.BATCH_SIZE, 1]).float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            count = i + 1
        avg_loss = running_loss / count
        return avg_loss

    def train_sequence(self,
                       train_loader,
                       test_loader,
                       directory,
                       epochs: int = env.EPOCHS):
        for epoch in range(epochs):
            self.train(True)
            t_loss = self.train_one_epoch(train_loader)
            self.train(False)
            v_loss = self.test_validation_sequence(test_loader, report=False)
            self.train_report.loc[epoch] = [epoch, t_loss, v_loss]
            typer.secho('Finished epoch # {:0>3} \t|\t '
                        'Train loss: {:0.6f} \t|\t Validation loss: {:0.6f}'.format(epoch+1, t_loss, v_loss),
                        fg="green")
        self.test_validation_sequence(test_loader, directory=directory, report=True)
        path = os.path.join(directory, 'train_report.csv')
        self.train_report.to_csv(path, index=False)
        self.train_report.plot(x='epoch', y=['train_loss', 'val_loss'], title='Training Report')
        plt.savefig(os.path.join(directory, 'training_report.png'))
        plt.close()

    def test_validation_sequence(self,
                                 test_loader,
                                 directory: os.path = None,
                                 report: bool = True):
        """
        The test sequence is used to test the model on the test set.
        :param directory:
        :param test_loader:
        :return:
        """
        if report:
            test_report = pd.DataFrame(columns=['real_angle', 'predicted_angle'])
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = self.criterion(output, target.reshape([env.BATCH_SIZE, 1]).float())
                if report:
                    df = np.stack([target.cpu().numpy(), output.squeeze().cpu().numpy()], axis=1)
                    test_report = test_report.append(pd.DataFrame(df, columns=['real_angle', 'predicted_angle']))
        if report:
            path = os.path.join(directory, 'test_report.csv')
            test_report.to_csv(path, index=False)
            return test_report
        else:
            return loss.detach().item()

    def save(self):
        """
        Saves the model to the directory specified in the environment file.
        :return:
        """
        path = os.path.join(env.MODELS_PATH, f'gauge_net_v{env.MODEL_VERSION}.pt')
        torch.save(self, path)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: sample of the input data (images of batch size)
        :return: predicted angle (in radians)
        """
        x = self.layers(x)
        return x.type(torch.float)

    @classmethod
    def load(cls,
             version: str = env.MODEL_VERSION):
        """
        Loads the model from the directory specified in the environment file
        :param version: optional, version of the model to load. if not specified, the default version is loaded
        :return: trained model from saved file
        """
        path = os.path.join(env.MODELS_PATH, f'gauge_net_v{version}.pt')
        return torch.load(path)
