import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple

# Beautiful high-contrast colors for each damage class (BGR format)
COLOR_PALETTE = {
    "scratch": (255, 255, 0),       # Cyan
    "dent": (255, 0, 255),          # Magenta
    "crack": (0, 255, 255),         # Yellow
    "rust": (0, 100, 255),          # Orange-red
    "glass_shatter": (255, 200, 0), # Light blue
    "broken_lamp": (0, 215, 255)    # Gold
}

def detect_calibration_factor(image_path: str) -> Tuple[float, Tuple[int, int, int, int] | None]:
    """
    Searches for a standard credit card reference object (aspect ratio ~1.58).
    If found, returns (px_to_cm_ratio, bounding_box).
    Otherwise returns (default_ratio, None).
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.05, None # 1 pixel = 0.05 cm default

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Standard credit card dimensions: 8.56 cm x 5.398 cm (ratio ~1.58)
    card_ratio = 8.56 / 5.398
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # We look for a 4-sided polygon
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            if bw < 50 or bh < 30: # Too small to be a reference card
                continue
                
            aspect_ratio = max(bw, bh) / float(min(bw, bh))
            # Check if aspect ratio is close to a credit card (~1.58)
            if 1.4 <= aspect_ratio <= 1.8:
                # Calculate pixels per cm based on the longer side (8.56 cm)
                pixels_per_cm = max(bw, bh) / 8.56
                cm_per_pixel = 1.0 / pixels_per_cm
                return cm_per_pixel, (x, y, bw, bh)
                
    # Default: 1 pixel = 0.04 cm (approx. 25 pixels per cm)
    return 0.04, None

def get_annotated_image_base64(image_path: str, detections: List[Dict[str, Any]], ref_box: Tuple[int, int, int, int] | None) -> str:
    """
    Draws bounding boxes, labels, and contours on the image and returns a base64 encoded string.
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""

    # Draw reference calibration box if detected
    if ref_box:
        rx, ry, rw, rh = ref_box
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(img, "Calibration Ref (Card)", (rx, max(15, ry - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw each damage detection
    for det in detections:
        x, y, w, h = det["box"]
        label = det["class"].replace("_", " ").title()
        conf = det["confidence"]
        color = COLOR_PALETTE.get(det["class"], (0, 255, 0))
        
        # Bounding Box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw contour if exists
        if "contour" in det and det["contour"]:
            pts = np.array(det["contour"], dtype=np.int32)
            cv2.drawContours(img, [pts], -1, color, 1)
            
        # Label Badge
        text = f"{label} ({int(conf * 100)}%)"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 10, y), color, -1)
        # Put text (contrast color: black for light badges)
        cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64
