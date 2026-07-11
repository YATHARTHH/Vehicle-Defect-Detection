"""
severity.py — Physical measurement and severity classification.

For dents: uses MiDaS depth estimator (falls back to formula).
For scratch/crack: uses bounding box diagonal as length proxy.
For rust: uses spread area.

If the YOLO model already encoded severity in the class name (e.g.
'moderate-dent'), that value is used directly and only measurements
are recalculated for display.
"""

from typing import Any

import cv2
import numpy as np
from services.depth_estimator import estimate_dent_depth_cm


class SeverityEstimator:
    def estimate(
        self,
        detections: list[dict[str, Any]],
        cm_per_pixel: float,
        image_path: str = "",
    ) -> list[dict[str, Any]]:
        """
        Enrich each detection with physical measurements and severity level.

        Parameters
        ----------
        detections   : Raw detections from DamageDetector.detect()
        cm_per_pixel : Calibration factor.
        image_path   : Path to original image (needed for MiDaS depth).
        """
        # Load image once for MiDaS crops
        img_bgr = cv2.imread(image_path) if image_path else None

        results = []
        for det in detections:
            x, y, w, h = det["box"]
            damage_class = det["class"]
            panel = det.get("panel", "Unknown Panel")
            preset_severity = det.get("preset_severity")  # from YOLO class name

            pixel_area = w * h
            cm2_area = round(pixel_area * (cm_per_pixel**2), 1)

            metrics: dict[str, Any] = {
                "pixel_area": pixel_area,
                "cm2_area": cm2_area,
            }

            # ── Measurements per damage class ─────────────────────────────
            if damage_class in ("scratch", "crack"):
                pixel_length = float(np.sqrt(w**2 + h**2))
                cm_length = round(pixel_length * cm_per_pixel, 1)
                metrics["length_cm"] = cm_length

                if preset_severity:
                    severity = preset_severity
                elif cm_length < 8.0:
                    severity = "Mild"
                elif cm_length < 25.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"

            elif damage_class == "dent":
                # Try MiDaS depth first; formula as fallback
                if img_bgr is not None:
                    cm_depth = estimate_dent_depth_cm(img_bgr, (x, y, w, h), cm_per_pixel)
                else:
                    diagonal_px = float(np.sqrt(w**2 + h**2))
                    cm_depth = max(0.2, round(diagonal_px * cm_per_pixel * 0.05, 1))

                metrics["depth_cm"] = cm_depth
                metrics["width_cm"] = round(w * cm_per_pixel, 1)
                metrics["height_cm"] = round(h * cm_per_pixel, 1)

                if preset_severity:
                    severity = preset_severity
                elif cm_depth < 0.8:
                    severity = "Mild"
                elif cm_depth < 2.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"

            elif damage_class == "rust":
                metrics["spread_area_cm2"] = cm2_area

                if preset_severity:
                    severity = preset_severity
                elif cm2_area < 12.0:
                    severity = "Mild"
                elif cm2_area < 50.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"

            elif damage_class in ("glass_shatter", "broken_lamp"):
                metrics["spread_area_cm2"] = cm2_area

                if preset_severity:
                    severity = preset_severity
                elif cm2_area < 20.0:
                    severity = "Mild"
                elif cm2_area < 75.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"

            else:
                severity = preset_severity or "Mild"

            results.append(
                {
                    "id": det["id"],
                    "class": damage_class,
                    "confidence": det["confidence"],
                    "box": det["box"],
                    "severity": severity,
                    "metrics": metrics,
                    "panel": panel,
                    "contour": det.get("contour", []),
                }
            )

        return results
