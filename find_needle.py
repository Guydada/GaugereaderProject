import cv2
import os
import numpy as np

import src.utils.envconfig as env


class FindNeedle:
    def __init__(self,
                 orig_cv,
                 directory: str):
        self.orig = orig_cv
        self.draw = False
        self.window_name = "Find the Needle APP"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 600)
        self.mask_image = np.zeros(self.orig.shape, dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.brush = [(255, 255, 255), -1]
        self.brush_size = 100  # initialize to minimum brush size #TODO: change to 2
        cv2.createTrackbar("Brush Size", self.window_name, self.brush_size, 100, self.nothing)
        cv2.createButton("Clear", self.clear_mask, cv2.QT_PUSH_BUTTON)
        cv2.setMouseCallback(self.window_name, self.draw_circle)
        wait_time = 1
        while cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(self.window_name, self.orig)
            key = cv2.waitKey(wait_time)
            code = key & 0xFF
            if key == ord('q') or key == 27 or key == ord('Q'):
                cv2.destroyAllWindows()
                break

        self.mask = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2GRAY)
        self.dst = cv2.inpaint(self.orig, self.mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(directory, 'train_gauge.jpg'), self.dst)
        self.needle = np.where(self.mask_image == 1, self.orig, self.mask_image)
        cv2.imwrite(os.path.join(directory, 'needle.jpg'), self.needle)

    def _set_brush(self, x, y):
        self.brush_size = cv2.getTrackbarPos('Brush Size', self.window_name)
        cv2.circle(self.orig, (x, y), self.brush_size, self.brush[0], self.brush[1])
        cv2.circle(self.mask_image, (x, y), self.brush_size, self.brush[0], self.brush[1])

    def clear_mask(self):
        self.mask_image = np.zeros(self.orig.shape, np.uint8)
        cv2.imshow(self.window_name, self.orig)

    def draw_circle(self,
                    event,
                    x,
                    y,
                    flags,
                    param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                self._set_brush(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False
            self._set_brush(x, y)

    def get_mask(self):
        return self.mask_image

    @staticmethod
    def nothing(x):
        pass


orig = cv2.imread(env.DEV_CALIBRATION_PHOTO_PATH.as_posix())
app = FindNeedle(orig, env.CALIBRATION_PATH)



