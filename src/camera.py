import cv2
from typing import Optional


class Camera:
    def __init__(self, index=0, width=416, height=312, rotate=0):
        self.index = index
        self.width = width
        self.height = height
        self.rotate = rotate  # degrees: 0, 90, 180, 270
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self):
        # Use cv2.CAP_ANY so OpenCV chooses appropriate backend
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        # set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

    def read(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.rotate:
            # rotate by 90-degree multiples
            k = (self.rotate % 360) // 90
            if k == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif k == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif k == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
