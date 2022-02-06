import cv2
import os
import numpy as np
import pandas as pd
import typer
import torch
import torchvision.transforms as tf
import PIL.Image as Image

import src.utils.image_editing as ie


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


scale = list(np.linspace(0.81, 0.99, 10))

image_augmentor = tf.Compose([tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5),
                              tf.RandomApply([tf.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
                              tf.RandomApply([tf.RandomPerspective(distortion_scale=np.random.choice(scale), p=0.5)], p=0.5),
                              tf.RandomApply([tf.RandomSolarize(threshold=[200, 0.5])], p=0.5)])


def create_train_test_set(train_image: np.ndarray,
                     needle_image: np.ndarray,
                     needle_center: tuple,
                     train_set_size: int,
                     test_set_size: int,
                     gauge_directory: str,
                     min_angle: float = 0,
                     max_angle: float = -360,
                     clear_previous_data: bool = False):
    # create directory for gauge images
    train_images_dir = os.path.join(gauge_directory, 'train_images')
    test_images_dir = os.path.join(gauge_directory, 'test_images')
    if os.path.exists(train_images_dir):
        if clear_previous_data:
            os.remove(train_images_dir)
    else:
        os.mkdir(train_images_dir)
    if os.path.exists(test_images_dir):
        if clear_previous_data:
            os.remove(test_images_dir)
    else:
        os.makedirs(test_images_dir)

    cols = ['image_name', 'augmented', 'angle']
    train_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)
    angles = np.random.uniform(min_angle, max_angle, train_set_size + test_set_size)
    angles = np.round(angles, 2)
    train_angles = np.random.choice(angles, train_set_size, replace=False)
    test_angles = np.array([angle for angle in angles if angle not in train_angles])
    for angle in train_angles:
        image_name = f"{angle:.2f}".replace(".", "_") + ".jpg"
        image = ie.rotate_needle(train_image, needle_image, needle_center, angle)
        image_pil = Image.fromarray(image)
        image_pil.save(f"{gauge_directory}/train/clean/{image_name}")
        train_df = train_df.append({'image_name': image_name, 'augmented': False, 'angle': angle}, ignore_index=True)
        image_name = f"{angle:.2f}".replace(".", "_") + "_augmented.jpg"
        image_pil = image_augmentor(image_pil)
        image_pil.save(f"{gauge_directory}/train/augmented/{image_name}")
        train_df = train_df.append({'image_name': image_name, 'augmented': True, 'angle': angle}, ignore_index=True)


    for angle in train_angles:
    typer.secho(f"{i} train images created", fg="green")
    i = 0
    for angle in test_angles:
        image_name = f"{angle:.2f}".replace(".", "_") + ".jpg"
        image = ie.rotate_needle(train_image, needle_image, needle_center, angle)
        cv2.imwrite(f"{gauge_directory}/validation/{image_name}", image)
        i += 1
    typer.secho(f"{i} validation images created", fg="green")


