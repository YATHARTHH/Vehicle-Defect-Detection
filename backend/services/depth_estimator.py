"""
depth_estimator.py — Monocular depth estimation for dent depth measurement.

Uses Intel MiDaS v2.1 (small variant) via PyTorch Hub.
- Downloads once (~90MB) on first call, cached in ~/.cache/torch/hub
- CPU inference: ~1-3 seconds per crop
- Falls back to bounding-box diagonal formula if torch/MiDaS unavailable.
"""

import os

import numpy as np

# Lazy-loaded globals — only initialise when first needed
_midas_model = None
_midas_transform = None
_midas_available: bool | None = None  # None = not yet checked


import logging

logger = logging.getLogger("overbody_api.depth_estimator")

def _try_load_midas() -> bool:
    """Attempt to load MiDaS small. Returns True on success."""
    global _midas_model, _midas_transform, _midas_available

    if _midas_available is not None:
        return _midas_available

    try:
        import torch
        import torchvision.transforms as T  # noqa – just verify it's present

        # Programmatically trust dependency repo to avoid console prompts
        try:
            trusted_file = os.path.expanduser("~/.cache/torch/hub/trusted_list")
            os.makedirs(os.path.dirname(trusted_file), exist_ok=True)
            lines = []
            if os.path.exists(trusted_file):
                with open(trusted_file) as f:
                    lines = [l.strip() for l in f.readlines()]
            if "rwightman_gen-efficientnet-pytorch" not in lines:
                with open(trusted_file, "a") as f:
                    f.write("rwightman_gen-efficientnet-pytorch\n")
        except Exception as te:
            logger.warning(f"Failed to update PyTorch Hub trusted list: {te}")

        model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            verbose=False,
            trust_repo=True,
        )
        model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            verbose=False,
            trust_repo=True,
        )
        transform = midas_transforms.small_transform

        _midas_model = model
        _midas_transform = transform
        _midas_available = True
        logger.info("MiDaS loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"MiDaS not available — using formula fallback. ({e})")
        _midas_available = False
        return False


def estimate_dent_depth_cm(
    image_bgr: np.ndarray,
    box_xywh: tuple,
    cm_per_pixel: float,
) -> float:
    """
    Estimate the physical depth of a dent in centimetres.

    Parameters
    ----------
    image_bgr   : Full image as a BGR numpy array (from cv2.imread).
    box_xywh    : Bounding box (x, y, w, h) of the dent region.
    cm_per_pixel: Calibration factor (cm per pixel).

    Returns
    -------
    Depth in centimetres (float, >= 0.1).
    """
    x, y, w, h = box_xywh
    ih, iw = image_bgr.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return _formula_depth(w, h, cm_per_pixel)

    if _try_load_midas():
        depth_cm = _midas_depth(roi, w, h, cm_per_pixel)
    else:
        depth_cm = _formula_depth(w, h, cm_per_pixel)

    return max(0.1, round(depth_cm, 1))


def _midas_depth(roi_bgr: np.ndarray, bw: int, bh: int, cm_per_pixel: float) -> float:
    """Run MiDaS on the dent crop and estimate physical depth."""
    import cv2
    import torch

    # MiDaS expects RGB
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    input_batch = _midas_transform(roi_rgb)

    with torch.no_grad():
        prediction = _midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=roi_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map to [0, 1] to keep range independent of raw prediction scales
    d_min_val = depth_map.min()
    d_max_val = depth_map.max()
    d_denom = (d_max_val - d_min_val) if (d_max_val - d_min_val) > 0 else 1.0
    depth_norm = (depth_map - d_min_val) / d_denom

    # Depth range within the crop (percentiles to avoid outliers)
    d_max = float(np.percentile(depth_norm, 95))
    d_min = float(np.percentile(depth_norm, 5))
    relative_range = d_max - d_min  # now guaranteed to be 0.0 - 1.0

    # Physical depth is a fraction of the bounding box's maximum physical dimension
    max_dim_px = max(bw, bh)
    max_dim_cm = max_dim_px * cm_per_pixel
    depth_cm = relative_range * max_dim_cm * 0.12  # approx. 12% max depth scaling

    return depth_cm


def _formula_depth(bw: int, bh: int, cm_per_pixel: float) -> float:
    """Bounding-box diagonal formula fallback."""
    diagonal_px = np.sqrt(bw**2 + bh**2)
    return diagonal_px * cm_per_pixel * 0.05
