import torch
import torch.nn as nn

import src.utils.image_editing as ie
import src.utils.envconfig as env


def test_model(model,
               test_loader,
               optimizer,
               criterion,
               epochs,
               device=env.DEVICE,
               plot: bool = False):
    """
    Test the model on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
