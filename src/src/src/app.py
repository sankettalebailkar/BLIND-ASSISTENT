#!/usr/bin/env python3
"""
Main loop for Blind Person Assistant (headless).
Reads frames from camera, runs YOLO detection, uses edge fallback,
estimates distance, and makes voice announcements via TTS.

Designed to be simple and easy to extend.
"""
import time
import argparse
import yaml
from collections import deque

from src.camera import Camera
from src.detector import Detector
from src.edge_detector import EdgeDetector
from src.distance import DistanceEstimator
from src.audio.tts import TTS


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    # camera config
    cam_cfg = cfg.get("camera", {})
    cam = Camera(
        index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 416),
        height=cam_cfg.get("height", 312),
        rotate=cam_cfg.get("rotate_deg", 0),
    )

    # detector
    det_cfg = cfg.get("detector", {})
    detector = Detector(
        model_path=det_cfg.get("model_path", "yolov8n.pt"),
        conf=det_cfg.get("conf_threshold", 0.35),
        iou=det_cfg.get("iou_threshold", 0.45),
        labels_map=det_cfg.get("labels_map", {}),
    )

    # edge fallback
    edge_cfg = cfg.get("edge_fallback", {})
    edge = EdgeDetector(
        center_ratio=edge_cfg.get("center_region_ratio", 0.35),
        density_threshold=edge_cfg.get("edge_density_threshold", 0.08),
    )

    # distance estimator
    dist_cfg = cfg.get("distance", {})
    dist = DistanceEstimator(
        focal_length=dist_cfg.get("focal_length", 700),
        known_width=dist_cfg.get("known_object_width", 0.5),
    )

    # tts
    general = cfg.get("general", {})
    tts = TTS(rate=general.get("tts_rate", 150))

    # announcement control
    announce_cfg = cfg.get("announce", {})
    min_distance_m = announce_cfg.get("min_distance_m", 2.1)
    avoid_repeat_seconds = announce_cfg.get("avoid_repeat_seconds", 3)

    announced_cache = {}  # msg -> last_time_announced
    processed_frames = 0

    try:
        cam.open()
        detector.load()
        print("System started. Press Ctrl+C to stop.")

        while True:
            loop_start = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # Run detection
            detections = detector.predict(frame)  # list sorted by confidence desc

            announced_msg = None
            if detections:
                # pick the most confident detection
                top = detections[0]
                label = top["label"]
                conf = top["confidence"]
                x1, y1, x2, y2 = top["box"]
                pixel_width = max(1.0, x2 - x1)
                est_m = dist.estimate_from_pixel_width(pixel_width)

                # form message
                announced_msg = f"{label} ahead, {est_m:.1f} meters"

                # decide whether to announce (distance threshold)
                if est_m <= min_distance_m:
                    now = time.time()
                    last = announced_cache.get(announced_msg, 0)
                    if now - last >= avoid_repeat_seconds:
                        tts.say(announced_msg)
                        announced_cache[announced_msg] = now

            else:
                # fallback: edge-based blocking detection
                if edge.is_blocking(frame):
                    announced_msg = "Obstacle ahead"
                    now = time.time()
                    last = announced_cache.get(announced_msg, 0)
                    if now - last >= avoid_repeat_seconds:
                        tts.say(announced_msg)
                        announced_cache[announced_msg] = now
                else:
                    # Optional: could announce "Path clear" but avoided by default
                    pass

            processed_frames += 1

            # throttle to target fps
            fps_target = general.get("fps_target", 10)
            elapsed = time.time() - loop_start
            target_frame_time = 1.0 / float(max(1, fps_target))
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    except KeyboardInterrupt:
        print("Keyboard interrupt received — stopping.")
    except Exception as e:
        print(f"Unhandled exception: {e}")
    finally:
        try:
            cam.close()
        except Exception:
            pass
        try:
            tts.stop()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blind Person Assistant main")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
