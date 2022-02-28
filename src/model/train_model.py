import torch
import torch.nn as nn

import src.utils.image_editing as ie
import src.utils.envconfig as env


def train_model(model,
                train_loader,
                optimizer,
                criterion,
                epochs,
                device=env.DEVICE,
                plot: bool = False):
    """

    :param plot:
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param epochs:
    :param device:
    :return:
    """
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, angles) in enumerate(train_loader):
            images = images.to(device)
            angles = angles.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, angles.reshape([env.BATCH_SIZE, 1]).float())
            loss.backward()
            optimizer.step()

            # def closure():
            #     optimizer.zero_grad()
            #     output = model(images)
            #     loss = criterion(output, angles.reshape([64, 1]).float())
            #     loss.backward()
            #     return loss
            # optimizer.step(closure)
            # loss = closure()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs} Loss: {running_loss:.5e}')

    return model
