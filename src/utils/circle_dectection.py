import cv2
import src.utils.point_math as math


def find_circles(image,
                 min_radius: int,
                 max_radius: int,
                 min_distance: int):
    """
    Finds circles in an image.
    :param image:
    :param min_radius:
    :param max_radius:
    :param min_distance:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=min_distance,
                               circles=1,
                               maxRadius=min_radius,
                               minRadius=max_radius)
    try:
        a, b, c = circles.shape
    except ValueError:
        return False

    x, y, r = math.avg_circles(circles, b)
    return x, y, r
