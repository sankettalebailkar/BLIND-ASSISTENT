import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, center_ratio=0.35, density_threshold=0.08):
        self.center_ratio = float(center_ratio)
        self.density_threshold = float(density_threshold)

    def is_blocking(self, frame) -> bool:
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        ch = max(1, int(h * self.center_ratio))
        cw = max(1, int(w * self.center_ratio))
        r1 = (h - ch) // 2
        c1 = (w - cw) // 2
        center = edges[r1:r1+ch, c1:c1+cw]
        if center.size == 0:
            return False
        edge_pixels = (center > 0).sum()
        density = edge_pixels / float(center.size)
        return density >= self.density_threshold
