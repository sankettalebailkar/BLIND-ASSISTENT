from ultralytics import YOLO
import numpy as np


class Detector:
    """
    Simple wrapper for ultralytics YOLO.
    predict(frame) -> list of dicts: {"label","confidence","box":[x1,y1,x2,y2]}
    """
    def __init__(self, model_path="yolov8n.pt", conf=0.35, iou=0.45, labels_map=None):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.labels_map = labels_map or {}
        self.model = None

    def load(self):
        # loads model into memory
        self.model = YOLO(self.model_path)

    def predict(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        # Ultralytics accept numpy BGR images directly
        res_list = self.model.predict(source=frame, conf=self.conf, iou=self.iou, verbose=False)
        if not res_list:
            return []

        r = res_list[0]
        detections = []
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            for box in r.boxes:
                # box.conf and box.cls may be tensors; convert safely
                try:
                    conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
                except Exception:
                    conf = float(box.conf) if box.conf is not None else 0.0

                try:
                    cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                except Exception:
                    cls_idx = int(box.cls)

                label = self.model.names.get(cls_idx, str(cls_idx))
                label = self.labels_map.get(label, label)

                # xyxy may be on CPU tensor or numpy
                try:
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
                except Exception:
                    try:
                        xyxy = np.array(box.xyxy)
                    except Exception:
                        continue

                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

            # sort descending by confidence
            detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
