import cv2
import numpy as np
import PIL.Image as Image
import typer


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
                  return_needle:bool = False):
    rotated_needle = rotate_image2(needle_image, needle_angle, needle_center)
    mask = cv2.cvtColor(rotated_needle, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    bg_img = cv2.bitwise_and(train_image, train_image, mask=mask_inv)
    blended = cv2.add(bg_img, rotated_needle)
    if return_needle:
        return rotated_needle
    return blended









