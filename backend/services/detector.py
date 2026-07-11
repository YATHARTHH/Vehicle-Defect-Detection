"""
detector.py — Vehicle surface damage detection using YOLOv8.

Primary path  : ultralytics YOLOv8 with a pretrained car-damage model.
                Model: abdullahg7/cardd-yolov8s (HuggingFace Hub)
                Downloaded and cached using huggingface_hub.

Fallback path : OpenCV-based heuristic detector.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import logging

logger = logging.getLogger("overbody_api.detector")

# ── Constants ─────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.35

# Map model classes -> internal frontend classes
CLASS_MAP: dict[str, str] = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass_shatter": "glass_shatter",
    "lamp_broken": "broken_lamp",
    "tire_flat": "dent",  # map tire flat to dent/body deformity fallback
}

_yolo_model = None  # cached YOLO instance
_yolo_available: bool | None = None  # None = not checked yet


def _try_load_yolo() -> bool:
    """Attempt to load the pretrained car-damage YOLOv8 model from Hugging Face."""
    global _yolo_model, _yolo_available

    if _yolo_available is not None:
        return _yolo_available

    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        logger.info("Checking Hugging Face model cache for cardd-yolov8s...")
        model_path = hf_hub_download(repo_id="abdullahg7/cardd-yolov8s", filename="v2.0/best.pt")
        model = YOLO(model_path)
        _yolo_model = model
        _yolo_available = True
        logger.info("YOLOv8 car-damage model loaded successfully from cache/HF.")
        return True

    except Exception as e:
        logger.error(f"YOLOv8 load failed — using OpenCV fallback. ({e})")
        _yolo_available = False
        return False


# ── Public detector class ─────────────────────────────────────────────────────


class DamageDetector:
    def __init__(self):
        # Load YOLO model at startup
        _try_load_yolo()

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        # Ensure we try to load model if not checked yet
        if _yolo_available is None:
            _try_load_yolo()

        if _yolo_available:
            return _yolo_detect(image_path)
        return _opencv_detect(image_path)


# ── YOLOv8 detection path ─────────────────────────────────────────────────────


def _yolo_detect(image_path: str) -> list[dict[str, Any]]:
    results = _yolo_model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        iou=0.45,
        verbose=False,
    )

    img = cv2.imread(image_path)
    if img is None:
        return []
    ih, iw = img.shape[:2]

    detections: list[dict[str, Any]] = []
    damage_id = 1

    result = results[0]
    names = result.names

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_idx = int(box.cls[0])
        raw_class = names[cls_idx].lower()

        internal_class = CLASS_MAP.get(raw_class, "dent")

        # xyxy -> xywh
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        bx = max(0, x1)
        by = max(0, y1)
        bw = min(iw, x2) - bx
        bh = min(ih, y2) - by

        if bw <= 0 or bh <= 0:
            continue

        panel = _classify_panel(bx, by, bw, bh, iw, ih)

        detections.append(
            {
                "id": damage_id,
                "class": internal_class,
                "raw_class": raw_class,
                "confidence": round(conf, 2),
                "box": [bx, by, bw, bh],
                "contour": [],
                "panel": panel,
                "preset_severity": None,
            }
        )
        damage_id += 1

    return detections


# ── OpenCV fallback ───────────────────────────────────────────────────────────


def _classify_panel(x: int, y: int, bw: int, bh: int, iw: int, ih: int) -> str:
    cx = x + bw / 2
    cy = y + bh / 2
    if cy < ih * 0.35:
        return "Hood / Roof"
    if cy > ih * 0.75:
        return "Lower Rocker Panel"
    if cx < iw * 0.25:
        return "Front Bumper / Fender"
    if cx > iw * 0.75:
        return "Rear Bumper / Trunk"
    return "Door Panel"


def _build_car_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    road_mask = (hsv[:, :, 2] < 60).astype(np.uint8) * 255
    grass_lo = np.array([35, 40, 40])
    grass_hi = np.array([90, 255, 255])
    grass_mask = cv2.inRange(hsv, grass_lo, grass_hi)

    car_mask = np.ones((h, w), dtype=np.uint8) * 255
    car_mask[road_mask > 0] = 0
    car_mask[grass_mask > 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, k)
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, k)
    return car_mask


def _opencv_detect(image_path: str) -> list[dict[str, Any]]:
    """Heuristic fallback detector — used when YOLOv8 is unavailable."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    car_mask = _build_car_mask(img)

    # Rust
    lo1 = np.array([0, 50, 40])
    hi1 = np.array([18, 255, 220])
    lo2 = np.array([160, 50, 40])
    hi2 = np.array([180, 255, 220])
    rust_mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    rust_mask = cv2.bitwise_and(rust_mask, car_mask)

    # Gradient map (dent surface)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.normalize(np.sqrt(sx**2 + sy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag = cv2.bitwise_and(mag, car_mask)
    _, mag_thresh = cv2.threshold(mag, 18, 255, cv2.THRESH_BINARY)

    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    deform = cv2.dilate(mag_thresh, k_large, iterations=3)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    deform = cv2.erode(deform, k_small, iterations=1)

    # Edges (scratches)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edges = cv2.bitwise_and(edges, car_mask)
    k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k_e, iterations=2)

    combined = cv2.bitwise_or(deform, edges)
    combined = cv2.bitwise_or(combined, rust_mask)
    combined = cv2.bitwise_and(combined, car_mask)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detections: list[dict[str, Any]] = []
    used: list[tuple[int, int, int, int]] = []
    damage_id = 1
    MARGIN = 6

    for contour in contours:
        if len(detections) >= 6:
            break
        area = cv2.contourArea(contour)
        if area < 400:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx <= MARGIN or by <= MARGIN:
            continue
        if (bx + bw) >= w - MARGIN or (by + bh) >= h - MARGIN:
            continue

        # De-duplicate
        dup = False
        for rx, ry, rw, rh in used:
            ox = max(0, min(bx + bw, rx + rw) - max(bx, rx))
            oy = max(0, min(by + bh, ry + rh) - max(by, ry))
            if ox * oy > 0.5 * min(bw * bh, rw * rh):
                dup = True
                break
        if dup:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        aspect = bw / float(bh) if bh > 0 else 1.0
        roi_gray = gray[by : by + bh, bx : bx + bw]
        roi_rust = rust_mask[by : by + bh, bx : bx + bw]
        roi_def = deform[by : by + bh, bx : bx + bw]

        rust_den = np.sum(roi_rust > 0) / float(bw * bh) if bw * bh > 0 else 0
        def_den = np.sum(roi_def > 0) / float(bw * bh) if bw * bh > 0 else 0
        std = float(np.std(roi_gray)) if roi_gray.size > 0 else 0

        if rust_den > 0.08:
            cls = "rust"
            conf = min(0.92, 0.65 + rust_den * 1.2)
        elif def_den > 0.4 and circularity > 0.25 and area > 2000:
            cls = "dent"
            conf = min(0.92, 0.72 + def_den * 0.2)
        elif aspect > 4.5 or aspect < 0.22:
            cls = "scratch"
            conf = min(0.90, 0.70 + min(0.18, area / 12000))
        elif circularity < 0.12 and area > 600:
            cls = "crack"
            conf = min(0.88, 0.65 + (0.12 - circularity) * 1.5)
        elif std > 35 and area > 1500:
            cls = "broken_lamp"
            conf = 0.74
        else:
            cls = "dent" if area > 4000 else "scratch"
            conf = 0.68

        panel = _classify_panel(bx, by, bw, bh, w, h)
        detections.append(
            {
                "id": damage_id,
                "class": cls,
                "raw_class": cls,
                "confidence": round(conf, 2),
                "box": [bx, by, bw, bh],
                "contour": contour.reshape(-1, 2).tolist(),
                "panel": panel,
                "preset_severity": None,
            }
        )
        used.append((bx, by, bw, bh))
        damage_id += 1

    return detections
