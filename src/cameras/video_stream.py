import cv2


class Capture:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path, src=0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.frame_width, self.frame_height)
        self.frame_shape = (self.frame_height, self.frame_width, 3)
        self.frame_dtype = "uint8"

    def start(self):
        self.cap.open(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Video stream not opened")

    def stop(self):
        self.cap.release()


if __name__ == '__main__':
    st = Capture("../../data/videos/test.mp4", 0)