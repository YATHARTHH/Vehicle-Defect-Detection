"""
restorer.py — Local Paint Restoration Engine.

Uses OpenCV's inpainting algorithms (Navier-Stokes and Fast Marching Method)
to visually erase detected vehicle exterior damages (scratches, dents, rust, etc.) 
and generate a photorealistic clean view.
"""

import cv2
import numpy as np
import base64
import logging
from typing import List, Dict, Any

logger = logging.getLogger("overbody_api.restorer")

class DamageRestorer:
    def __init__(self):
        pass

    def restore(self, image_path: str, damages: List[Dict[str, Any]]) -> str:
        """
        Creates a paint restoration mask based on detected damage coordinates/contours,
        runs inpainting, and returns the restored image as a base64 JPEG string.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"[restorer] Failed to load image from path: {image_path}")
            return ""

        # If there are no damages, return the original image encoded in base64
        if not damages:
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')

        # 1. Create a binary single-channel mask (same width and height as original)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Draw damage areas on the mask (white = regions to restore)
        for det in damages:
            # Prefer precise contour if available
            if "contour" in det and det["contour"]:
                try:
                    pts = np.array(det["contour"], dtype=np.int32)
                    # Draw a filled polygon on the mask
                    cv2.drawContours(mask, [pts], -1, 255, -1)
                    continue
                except Exception as ce:
                    logger.warning(f"[restorer] Failed to parse contour for inpainting: {ce}")

            # Fall back to bounding box if contour is missing or failed
            if "box" in det and det["box"]:
                bx, by, bw, bh = det["box"]
                # Draw a filled rectangle on the mask
                cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

        # 3. Dilate the mask slightly (e.g. 5x5 kernel) so we repaint slightly past the boundaries
        # This ensures smooth blending with the surrounding healthy paint texture.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 4. Perform OpenCV inpainting (using Fast Marching Method by Alexandru Telea)
        # Inpaint radius: 5 pixels is ideal for typical scratches/dents
        try:
            logger.info(f"[restorer] Running inpainting on {len(damages)} damage regions...")
            restored_img = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        except Exception as ie:
            logger.error(f"[restorer] Inpainting failed, falling back to original: {ie}")
            restored_img = img

        # 5. Encode the restored image to base64 JPEG
        _, buffer = cv2.imencode('.jpg', restored_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
