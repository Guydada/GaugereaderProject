import cv2
import typer
import torch
import numpy as np
from PIL import ImageFilter
import PIL.Image as Image
import torchvision.transforms as tf
import src.utils.envconfig as env


def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderMode=cv2.BORDER_TRANSPARENT,)
    return result


def rotate_image2(img, angle, pivot):
    img = Image.fromarray(img)
    rotated = img.rotate(angle, center=pivot)
    return np.array(rotated)


def rotate_needle(train_image: np.ndarray,
                  needle_image: np.ndarray,
                  needle_center: tuple,
                  needle_angle: float,
                  return_needle: bool = False):
    rotated_needle = rotate_image2(needle_image, needle_angle, needle_center)
    mask = cv2.cvtColor(rotated_needle, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    bg_img = cv2.bitwise_and(train_image, train_image, mask=mask_inv)
    blended = cv2.add(bg_img, rotated_needle)
    if return_needle:
        return rotated_needle
    return blended


scale = list(np.linspace(0.81, 0.99, 10))

process_image = tf.Compose([tf.Resize(env.TRAIN_IMAGE_SIZE),
                            tf.ToTensor(),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image_augmentor = tf.Compose([tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5),
                              tf.RandomApply([tf.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.1),
                              tf.RandomApply([tf.RandomPerspective(distortion_scale=np.random.choice(scale), p=0.5)], p=0.5)])










