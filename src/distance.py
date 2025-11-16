class DistanceEstimator:
    """
    Estimate distance using formula:
        distance = (known_width * focal_length) / pixel_width
    focal_length should be calibrated ahead of time.
    """
    def __init__(self, focal_length=700.0, known_width=0.5):
        self.focal_length = float(focal_length)
        self.known_width = float(known_width)

    def estimate_from_pixel_width(self, pixel_width: float) -> float:
        try:
            pw = float(pixel_width)
            if pw <= 0:
                return 999.0
            return float((self.known_width * self.focal_length) / pw)
        except Exception:
            return 999.0
