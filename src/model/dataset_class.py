import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        return image_path, label

    def __len__(self):
        return len(self.image_paths)
