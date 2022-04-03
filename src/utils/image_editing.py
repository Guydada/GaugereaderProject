import cv2
import os
from dataclasses import dataclass, asdict
import numpy as np
import PIL.Image as Image
import PIL.ImageTk as ImageTk
import torchvision.transforms as tf

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


def rotate_image(img,
                 angle,
                 pivot=None):
    img = Image.fromarray(img)
    rotated = img.rotate(angle, center=pivot)
    return np.array(rotated)


def rotate_needle(train_image: np.ndarray,
                  needle_image: np.ndarray,
                  needle_center: tuple,
                  needle_angle: float):
    rotated_needle = rotate_image(needle_image, needle_angle, needle_center)
    mask = cv2.cvtColor(rotated_needle, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    bg_img = cv2.bitwise_and(train_image, train_image, mask=mask_inv)
    blended = cv2.add(bg_img, rotated_needle)
    return blended, rotated_needle


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


def four_point_transform(image,
                         pts: list):
    """
    Apply a perspective transform to a frame, given the 4 points that define the
    transformation. Returns the transformed frame and the perspective transform ordered points.
    :param pts:
    :param image: cv2 image
    :param pts: points that define the transformation
    :return: warped image and ordered points
    """
    h, w = image.shape[:2]
    source = np.float32([[0, 0],
                         [w, 0],
                         [w, h],
                         [0, h]])
    dest = np.float32(pts)
    M = cv2.getPerspectiveTransform(dest, source)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


def frame_to_read_image(frame,
                        crop_coords,
                        perspective_pts,
                        perspective_changed: bool = False):
    """
    Convert a frame to a PIL image, crop it, and apply a perspective transform
    :param perspective_changed:
    :param crop_coords:
    :param perspective_pts:
    :param frame:
    :return:
    """
    y, y_diff, x, x_diff = crop_coords
    frame = frame[y:y + y_diff, x:x + x_diff]
    if perspective_changed:
        frame = four_point_transform(frame, perspective_pts)
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, env.TRAIN_IMAGE_SHAPE)
    cv2.imwrite("edited_frame.png", frame)
    frame = Image.fromarray(frame)
    transform = tf.Compose([tf.ToTensor(), tf.Normalize(mean=[0.5], std=[0.5])])
    frame = transform(frame)
    return frame.unsqueeze(0).to(env.DEVICE)


scale = list(np.linspace(0.81, 0.99, 10))

process_image = tf.Compose([tf.ToTensor(),
                            tf.Normalize(mean=[0.5], std=[0.5])])

image_augmentor = tf.Compose([tf.RandomApply([tf.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5)])


@dataclass
class Perspective:
    tl_x: int = 0
    tl_y: int = 0
    tr_x: int = 0
    tr_y: int = 0
    br_x: int = 0
    br_y: int = 0
    bl_x: int = 0
    bl_y: int = 0

    def __post_init__(self):
        self.point_names = list(self.asdict().keys())
        self.points = self.get_points()
        self.changed = False
        self.draw = []

    def asdict(self):
        return asdict(self)

    def aslist(self):
        return list(self.asdict().values())

    def get_points(self):
        lst = self.aslist()
        points = np.array(lst).reshape(4, 2).tolist()
        return points

    def set_points(self,
                   points: list = None,
                   order: bool = True):
        if points is None:
            points = self.draw
        if len(points) != 4:
            points = np.array(points).reshape(4, 2).tolist()
        if order:
            points = pm.order_points(points)
        for i, key in enumerate(self.point_names):
            j = i // 2
            if key[-1] == 'x':
                setattr(self, key, points[j][0])
            else:
                setattr(self, key, points[j][1])
        self.points = points
        self.changed = True

    def reset(self,
              w: int,
              h: int = None):
        h = w if h is None else h
        self.tl_x = 0
        self.tl_y = 0
        self.tr_x = w
        self.tr_y = 0
        self.br_x = w
        self.br_y = h
        self.bl_x = 0
        self.bl_y = h

    def delete_draw(self):
        self.draw = []

    def __getitem__(self, item):
        return self.asdict()[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.points = self.get_points()
