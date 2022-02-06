import cv2


def get_frame(stream):
    # read frame from video stream
    frame = stream.read()
    # convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # return frame
    return frame