import os
import time
import torch
import typer
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import settings

torch.manual_seed(settings.TORCH_SEED)


class GaugeNet(nn.Module):
    def __init__(self,
                 directory: str):
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
                                    nn.ReLU(),
                                    nn.Linear(in_features=512,
                                              out_features=256),
                                    nn.ReLU(),
                                    nn.Linear(in_features=256,
                                              out_features=128),
                                    nn.ReLU(),
                                    nn.Linear(in_features=128,
                                              out_features=1))

        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

        self.device = settings.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings.LEARNING_RATE)
        self.train_report = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
        self.best_epoch = 0
        self.best_loss = np.inf
        self.directory = directory

    def train_one_epoch(self,
                        train_loader):
        running_loss = 0.0
        for i, (images, angles) in enumerate(train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            self.optimizer.zero_grad()
            output = self(images)
            loss = self.criterion(output, angles.reshape([settings.BATCH_SIZE, 1]).float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            count = i + 1
        avg_loss = running_loss / count
        return avg_loss

    def train_sequence(self,
                       train_loader,
                       val_loader,
                       test_loader,
                       epochs: int = settings.EPOCHS,
                       epochs_add: int = settings.EPOCHS_ADD,
                       auto_add: bool = settings.AUTO_ADD_EPOCHS,
                       max_epochs: int = settings.MAX_EPOCHS,
                       transfer_learning: bool = True):
        if transfer_learning:
            try:
                self = self.load(directory=self.directory)
                typer.secho('Model loaded from checkpoint, transfer learning in progress', fg='yellow')
            except FileNotFoundError or TypeError:
                typer.secho('No checkpoint found, starting from scratch', fg='yellow')
        else:
            typer.secho('Starting from scratch. Existing weight files will be deleted', fg='yellow')

        epoch = 0
        thres = settings.LOSS_THRESHOLD

        typer.secho(f'Training initial epochs number: {epochs} '
                    f'| Max number of epochs: {max_epochs} '
                    f'| Automatically adding epochs is {"ON" if auto_add else "OFF"}', fg=typer.colors.CYAN)

        train_start_time = time.time()

        while epoch < epochs:
            epoch_start_time = time.time()
            self.train(True)
            t_loss = self.train_one_epoch(train_loader)
            self.train(False)
            v_loss = self.test_validation_sequence(val_loader, report=False)
            if v_loss <= self.best_loss:
                self.best_loss = v_loss
                self.best_epoch = epoch
                self.save(epoch='best')
            self.train_report.loc[epoch] = [epoch, t_loss, v_loss]
            check = '✔' if v_loss <= thres else '❌'
            epoch_time = time.time() - epoch_start_time

            typer.echo('Finished epoch # {:0>3} \t|\t '
                       'Train loss: {:0.6f} \t|\t Validation loss: {:0.6f}'
                       ' {} \t|\t '
                       'Epoch time: {:0.2f} MIN'.format(epoch + 1, t_loss, v_loss, check, epoch_time / 60))

            if all([auto_add,
                    epoch == epochs - 1,
                    self.best_loss > thres,
                    epoch < max_epochs - 1,
                    ],
                   ):
                epochs_add = min(epochs_add, max_epochs - epoch - 1)
                typer.secho(f'Validation loss below threshold, adding {epochs_add} epochs', fg='yellow')
                epochs += epochs_add
            elif not auto_add and epoch == epochs:
                break

            epoch += 1
        total_time = (time.time() - train_start_time) / 60
        typer.secho(f'Training finished in {total_time:0.2f} MIN', fg='yellow')

        self.save(epoch='last')
        """
        Last epoch is used to test the model on the test set and validation set.
        """
        val_loss = self.test_validation_sequence(val_loader, report=False)
        test_loss = self.test_validation_sequence(test_loader, report=False)
        self.print_loss(val_loss, test_loss)
        """
        Best epoch is used to test the model on the test set and validation set. Only the best model is saved is used 
        for the reports.
        """
        epoch = 'best'
        val_loss = self.test_validation_sequence(val_loader, report=True, epoch=epoch)
        test_loss = self.test_validation_sequence(test_loader, report=True, epoch=epoch, set_name='test')
        self.print_loss(val_loss, test_loss, epoch=epoch)

        path = os.path.join(self.directory, 'train_report.csv')
        self.train_report.to_csv(path, index=False)
        self.train_report.plot(x='epoch', y=['train_loss', 'val_loss'], title='Training Report')
        plt.savefig(os.path.join(self.directory, 'training_plots.png'))
        plt.close()

    def test_validation_sequence(self,
                                 test_loader,
                                 report: bool = True,
                                 set_name: str = 'val',
                                 epoch: str = 'last'):
        """
        The test and validation sequence is used to test the model on the test/validation set.
        :param test_loader:
        :return:
        """
        if report:
            df_report = pd.DataFrame(columns=['real_angle', 'predicted_angle'])
        if epoch == 'best':
            model = self.load(directory=self.directory, epoch=epoch)
        else:
            model = self
        with torch.no_grad():
            for data, target in test_loader:
                if isinstance(data, torch.Tensor) and isinstance(target, torch.Tensor):
                    data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target.reshape([settings.BATCH_SIZE, 1]).float())
                if report:
                    df_temp = np.stack([target.cpu().numpy(), output.squeeze().cpu().numpy()], axis=1)
                    df_report = pd.concat([df_report,
                                           pd.DataFrame(df_temp,
                                                        columns=['real_angle', 'predicted_angle'])])
        if report:
            path = os.path.join(self.directory, f'{set_name}_report.csv')
            df_report.to_csv(path, index=False)
        return loss.detach().item()

    def save(self,
             directory: str = None,
             epoch: str = None):
        """
        Saves the model to the directory specified in the environment file.
        :return:
        """
        if directory is None:
            directory = self.directory
        path = os.path.join(directory, f'gauge_net_v{settings.MODEL_VERSION}_{epoch}.pt')
        if not os.path.exists(directory):
            os.makedirs(settings.MODELS_PATH)
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
             directory: str = None,
             version: str = settings.MODEL_VERSION,
             epoch: str = settings.DEFAULT_MODEL_TYPE):
        """
        Loads the model from the directory specified in the environment file
        :param directory: Gauge directory
        :param version: optional, version of the model to load. if not specified, the default version is loaded
        :param epoch: optional, epoch of the model to load. if not specified, the best epoch is loaded
        :return: trained model from saved file
        """
        path = os.path.join(directory, f'gauge_net_v{version}_{epoch}.pt')
        return torch.load(path)

    @staticmethod
    def print_loss(val_loss: float,
                   test_loss: float,
                   epoch: str = 'last'):
        """
        Prints the loss of the model for the validation and test set.
        :param val_loss:
        :param test_loss:
        :param epoch: epoch version of the model: 'last', 'best'
        :return:
        """
        icon = '🧪' if epoch == 'last' else '💪'
        # icon = '' if epoch == 'last' else ''
        typer.echo(
            'Final loss using {} \t{}\t epoch model weights \t|\t Validation {:0.6f} \t|\t Test: {:0.6f}'.format(
                epoch.upper(),
                icon,
                val_loss,
                test_loss))
