#!/usr/bin/env python3
"""
run_pi5_dual_assistant.py

Final single-file code for Raspberry Pi 5 with dual cameras.

Features:
- Dual camera capture (two Pi cameras or two /dev/video* devices)
- YOLOv8 detection (ultralytics)
- Focus on important classes:
    - person, animal, dog, cat, cow
    - vehicle (car, bus, truck, motorbike, bicycle)
    - pothole
    - stairs
    - traffic light
    - blackboard (tv mapped)
    - pole, tree, bench, chair, desk, road sign, etc.
- Label mapping to clean, user-friendly labels
- Simple distance estimation from bounding-box width
- Edge-density fallback when no objects detected (detect walls/obstacles)
- Offline TTS (pyttsx3)
- Announcement only when object is within ~7 feet
- Avoids repeating same message too often
- Processes/announces every N frames
- Headless (no GUI)

Run:
    python3 run_pi5_dual_assistant.py
"""

import time
import threading
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# =========================
# CONFIG (EDIT THESE)
# =========================

# Path to your YOLOv8 model (.pt). Put your custom/COCO model here.
MODEL_PATH = "models/yolov8n.pt"

# Dual camera devices: use 0 & 1 (or "/dev/video0", "/dev/video1")
CAM_DEVICE_1 = 0
CAM_DEVICE_2 = 1

# Camera resolution
FRAME_WIDTH = 416
FRAME_HEIGHT = 312

# Cameras rotated 90° anti-clockwise -> 270° rotation
CAM_ROTATE_DEG = 270  # 0, 90, 180, 270

# YOLO thresholds
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# FPS target and frame processing rate
FPS_TARGET = 10
PROCESS_EVERY_N_FRAMES = 2  # process/announce every 2 frames

# Distance (announcement) threshold
# 7 feet ≈ 2.1336 meters
ANNOUNCE_DISTANCE_M = 2.1336

# Avoid repeating exact same sentence within this many seconds
AVOID_REPEAT_SECONDS = 3.0

# Edge detection fallback (for walls/unknown obstacles)
EDGE_CENTER_RATIO = 0.35
EDGE_DENSITY_THRESHOLD = 0.08

# Distance estimation parameters (calibrate FOCAL_LENGTH_PIXELS for your camera)
FOCAL_LENGTH_PIXELS = 700.0      # change after calibration
KNOWN_WIDTH_DEFAULT_M = 0.5      # generic default width (meters)

# TTS rate
TTS_RATE = 150

# Max number of detections to announce at once
MAX_DETECTIONS_TO_ANNOUNCE = 3

# =========================
# LABEL MAPPING & CLASSES
# =========================

# Map raw YOLO labels to user-friendly labels
LABELS_MAP = {
    # Vehicles
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",

    # Blackboard
    "tv": "blackboard",

    # Animals -> keep some separate, some generic
    "dog": "dog",
    "cat": "cat",
    "cow": "cow",
    "horse": "animal",
    "sheep": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    "bird": "animal",

    # Road/traffic
    "traffic light": "traffic light",
    "stop sign": "road sign",
    "parking meter": "road sign",
    "fire hydrant": "road sign",
}

# Important labels we care about (after mapping)
IMPORTANT_LABELS = {
    "person",
    "vehicle",
    "dog",
    "cat",
    "cow",
    "animal",
    "pothole",
    "stairs",
    "traffic light",
    "blackboard",
    "road sign",
    "pole",
    "tree",
    "bench",
    "chair",
    "desk",
    "table",
}

# Approximate real-world widths (meters) for distance estimation
CLASS_KNOWN_WIDTHS = {
    "person": 0.5,          # shoulder width
    "vehicle": 1.8,         # car width
    "cow": 1.5,
    "dog": 0.4,
    "cat": 0.2,
    "animal": 0.7,
    "pothole": 0.6,
    "stairs": 1.0,
    "traffic light": 0.3,
    "blackboard": 1.5,
    "road sign": 0.6,
    "pole": 0.2,
    "tree": 0.7,
    "bench": 1.2,
    "chair": 0.5,
    "desk": 1.0,
    "table": 1.0,
}

# =========================
# HELPER CLASSES
# =========================

class TTS:
    """Threaded offline TTS using pyttsx3."""

    def __init__(self, rate: int = TTS_RATE):
        self.engine = pyttsx3.init()
        try:
            self.engine.setProperty("rate", int(rate))
        except Exception:
            pass
        self._lock = threading.Lock()
        self._stopped = False

    def _speak_blocking(self, text: str):
        with self._lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                # Don't crash main loop on TTS error
                pass

    def say(self, text: str):
        if self._stopped:
            return
        th = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        th.start()

    def stop(self):
        self._stopped = True
        with self._lock:
            try:
                self.engine.stop()
            except Exception:
                pass


class DualCamera:
    """Manage two cameras with OpenCV."""

    def __init__(self, dev1, dev2, width=FRAME_WIDTH, height=FRAME_HEIGHT, rotate_deg=CAM_ROTATE_DEG):
        self.dev1 = dev1
        self.dev2 = dev2
        self.w = int(width)
        self.h = int(height)
        self.rotate = int(rotate_deg) % 360
        self.cap1: Optional[cv2.VideoCapture] = None
        self.cap2: Optional[cv2.VideoCapture] = None

    def open(self):
        self.cap1 = cv2.VideoCapture(self.dev1, cv2.CAP_ANY)
        self.cap2 = cv2.VideoCapture(self.dev2, cv2.CAP_ANY)

        if not self.cap1 or not self.cap1.isOpened():
            raise RuntimeError(f"Cannot open camera {self.dev1}")
        if not self.cap2 or not self.cap2.isOpened():
            raise RuntimeError(f"Cannot open camera {self.dev2}")

        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap2.set(cv2.CAP_PROP_FRAME_FRAME_WIDTH, self.w) if hasattr(cv2, "CAP_PROP_FRAME_FRAME_WIDTH") else None
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        k = (self.rotate // 90) % 4
        if k == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif k == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ok1, f1 = self.cap1.read() if self.cap1 else (False, None)
        ok2, f2 = self.cap2.read() if self.cap2 else (False, None)
        if not ok1:
            f1 = None
        if not ok2:
            f2 = None
        if f1 is not None and self.rotate:
            f1 = self._rotate(f1)
        if f2 is not None and self.rotate:
            f2 = self._rotate(f2)
        return f1, f2

    def release(self):
        try:
            if self.cap1:
                self.cap1.release()
        except Exception:
            pass
        try:
            if self.cap2:
                self.cap2.release()
        except Exception:
            pass


# =========================
# HELPER FUNCTIONS
# =========================

def map_label(label: str) -> str:
    """Map raw YOLO label to friendly label."""
    return LABELS_MAP.get(label, label)


def is_important(label: str) -> bool:
    """Return True if label is one of the important ones we care about."""
    return label in IMPORTANT_LABELS


def get_known_width(label: str) -> float:
    """Return real-world width for class, else default."""
    return CLASS_KNOWN_WIDTHS.get(label, KNOWN_WIDTH_DEFAULT_M)


def estimate_distance(focal_pixels: float, known_width_m: float, pixel_width: float) -> float:
    """Distance = (known_width * focal_length) / pixel_width."""
    if pixel_width <= 0 or focal_pixels <= 0:
        return float('inf')
    return (known_width_m * focal_pixels) / pixel_width


def edge_blocking(frame: np.ndarray,
                  center_ratio: float = EDGE_CENTER_RATIO,
                  density_thr: float = EDGE_DENSITY_THRESHOLD) -> bool:
    """Detect if central region has high edge density (likely wall/obstacle)."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        ch = max(1, int(h * center_ratio))
        cw = max(1, int(w * center_ratio))
        r1 = (h - ch) // 2
        c1 = (w - cw) // 2
        center = edges[r1:r1 + ch, c1:c1 + cw]
        if center.size == 0:
            return False
        edge_pixels = (center > 0).sum()
        density = edge_pixels / float(center.size)
        return density >= density_thr
    except Exception:
        return False


# =========================
# MAIN LOOP
# =========================

def main():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cam = DualCamera(
        CAM_DEVICE_1,
        CAM_DEVICE_2,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        rotate_deg=CAM_ROTATE_DEG,
    )
    tts = TTS(rate=TTS_RATE)
    announced_cache: Dict[str, float] = {}
    frame_count = 0

    try:
        cam.open()
        print("[INFO] Dual cameras opened successfully.")
        print("[INFO] Starting main detection + voice loop. Press Ctrl+C to stop.")

        while True:
            loop_start = time.time()
            f1, f2 = cam.read()

            frames: List[Tuple[str, np.ndarray]] = []
            if f1 is not None:
                frames.append(("left", f1))
            if f2 is not None:
                frames.append(("right", f2))

            detections_all: List[Dict] = []

            # Run YOLO on each available frame
            for cam_id, frame in frames:
                results = model.predict(
                    source=frame,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False,
                )
                if not results:
                    continue

                r = results[0]
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    for box in r.boxes:
                        # Confidence
                        try:
                            conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
                        except Exception:
                            try:
                                conf = float(box.conf)
                            except Exception:
                                conf = 0.0

                        # Class index
                        try:
                            cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                        except Exception:
                            try:
                                cls_idx = int(box.cls)
                            except Exception:
                                cls_idx = -1

                        raw_label = str(model.names.get(cls_idx, str(cls_idx)))
                        label = map_label(raw_label)

                        # Ignore non-important labels to reduce noise
                        if not is_important(label):
                            continue

                        # Bounding box
                        try:
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
                        except Exception:
                            try:
                                xyxy = np.array(box.xyxy)
                            except Exception:
                                continue

                        x1, y1, x2, y2 = map(float, xyxy)
                        pixel_w = max(1.0, x2 - x1)

                        known_w = get_known_width(label)
                        dist_m = estimate_distance(FOCAL_LENGTH_PIXELS, known_w, pixel_w)

                        detections_all.append({
                            "cam": cam_id,
                            "label": label,
                            "conf": conf,
                            "box": [x1, y1, x2, y2],
                            "pixel_width": pixel_w,
                            "est_m": dist_m,
                        })

            frame_count += 1
            do_process = (frame_count % PROCESS_EVERY_N_FRAMES) == 0

            if do_process:
                # Sort detections by distance (nearest first) then confidence
                detections_all.sort(key=lambda d: (d["est_m"], -d["conf"]))

                # Filter detections that are within announce distance
                close_dets = [
                    d for d in detections_all
                    if d["est_m"] <= ANNOUNCE_DISTANCE_M
                ]

                if close_dets:
                    # Announce up to MAX_DETECTIONS_TO_ANNOUNCE
                    now = time.time()
                    used_labels = set()
                    for det in close_dets[:MAX_DETECTIONS_TO_ANNOUNCE]:
                        label = det["label"]
                        est_m = det["est_m"]
                        cam_id = det["cam"]

                        # Avoid announcing same label multiple times if from both cameras
                        key_label = (label, cam_id)
                        if label in used_labels:
                            continue
                        used_labels.add(label)

                        msg = f"{label} ahead, {est_m:.1f} meters"
                        last = announced_cache.get(msg, 0.0)
                        if now - last >= AVOID_REPEAT_SECONDS:
                            print("[VOICE]", msg)
                            tts.say(msg)
                            announced_cache[msg] = now
                            time.sleep(0.2)  # tiny gap to avoid voice overlap
                else:
                    # No close detections: check edge fallback
                    blocked_any = False
                    for cam_id, frame in frames:
                        if edge_blocking(frame, center_ratio=EDGE_CENTER_RATIO, density_thr=EDGE_DENSITY_THRESHOLD):
                            blocked_any = True
                            break

                    if blocked_any:
                        msg = "Obstacle ahead"
                        now = time.time()
                        last = announced_cache.get(msg, 0.0)
                        if now - last >= AVOID_REPEAT_SECONDS:
                            print("[VOICE]", msg)
                            tts.say(msg)
                            announced_cache[msg] = now
                    else:
                        # Do not repeatedly say "Path clear" – keep silent here.
                        pass

                # Clean old cache entries to limit memory
                now = time.time()
                for k, t0 in list(announced_cache.items()):
                    if now - t0 > 60.0:
                        announced_cache.pop(k, None)

            # Throttle FPS
            elapsed = time.time() - loop_start
            target_dt = 1.0 / max(1, FPS_TARGET)
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, stopping.")
    except Exception as e:
        print("[ERROR] Fatal exception:", e)
    finally:
        print("[INFO] Cleaning up...")
        try:
            cam.release()
        except Exception:
            pass
        try:
            tts.stop()
        except Exception:
            pass
        print("[INFO] Exit complete.")


if __name__ == "__main__":
    main()
