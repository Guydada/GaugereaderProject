import cv2
import typer
import torch
import numpy as np
from PIL import ImageFilter
import PIL.Image as Image
import PIL.ImageTk as ImageTk
import torchvision.transforms as tf
import torchvision.transforms.functional as F

import src.utils.envconfig as env
import src.utils.point_math as pm


def cv_to_imagetk(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


def cv_to_image(cv_image, show: bool = False):
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    if show:
        image.show()
    return image


def rotate_image(img, angle, pivot):
    img = Image.fromarray(img)
    rotated = img.rotate(angle, center=pivot)
    return np.array(rotated)


def rotate_needle(train_image: np.ndarray,
                  needle_image: np.ndarray,
                  needle_center: tuple,
                  needle_angle: float,
                  return_needle: bool = False):
    rotated_needle = rotate_image(needle_image, needle_angle, needle_center)
    mask = cv2.cvtColor(rotated_needle, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    bg_img = cv2.bitwise_and(train_image, train_image, mask=mask_inv)
    blended = cv2.add(bg_img, rotated_needle)
    if return_needle:
        return rotated_needle
    return blended


def create_circle(obj,
                  x,
                  y,
                  r,
                  **kwargs):
    """
    Create a circle with the given parameters, implemented using the tkinter canvas drawing
    object method
    :param obj:
    :param x:
    :param y:
    :param r:
    :param kwargs:
    :return:
    """
    return obj.create_oval(x - r, y - r, x + r, y + r, **kwargs)


def four_point_transform(image, pts):
    h, w = image.shape[:2]
    source = np.float32([[0, 0],
              [w, 0],
              [w, h],
              [0, h]])
    pts = pm.order_points(pts)
    dest = np.float32(pts)
    M = cv2.getPerspectiveTransform(dest, source)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


def frame_to_read_image(frame,
                        crop_coords,
                        perspective_pts):
    """
    Convert a frame to a PIL image, crop it, and apply a perspective transform
    :param crop_coords:
    :param perspective_pts:
    :param frame:
    :return:
    """
    y, y_diff, x, x_diff = crop_coords
    frame = frame[y:y + y_diff, x:x + x_diff]
    frame = four_point_transform(frame, perspective_pts)
    frame = cv2.resize(frame, env.TRAIN_IMAGE_SIZE)
    return frame


scale = list(np.linspace(0.81, 0.99, 10))

process_image = tf.Compose([tf.Resize(env.TRAIN_IMAGE_SIZE),
                            tf.ToTensor(),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image_augmentor = tf.Compose([tf.RandomApply([tf.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5)])
# tf.RandomApply([tf.RandomRotation(0.5)], p=0.25)])
