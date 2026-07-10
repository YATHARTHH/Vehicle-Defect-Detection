import numpy as np
from typing import Dict, List, Any

class SeverityEstimator:
    def estimate(self, detections: List[Dict[str, Any]], cm_per_pixel: float) -> List[Dict[str, Any]]:
        """
        Estimates the physical size and classifies the severity level of each detection.
        """
        results = []
        for det in detections:
            x, y, w, h = det["box"]
            damage_class = det["class"]
            
            # 1. Bounding box area in cm^2
            pixel_area = w * h
            cm2_area = round(pixel_area * (cm_per_pixel ** 2), 1)
            
            # Initialize metrics dict
            metrics = {
                "pixel_area": pixel_area,
                "cm2_area": cm2_area
            }
            
            severity = "Mild"
            
            if damage_class in ["scratch", "crack"]:
                # Estimate length based on contour diagonal or bounding box diagonal
                pixel_length = np.sqrt(w**2 + h**2)
                cm_length = round(pixel_length * cm_per_pixel, 1)
                metrics["length_cm"] = cm_length
                
                # Classify severity
                if cm_length < 8.0:
                    severity = "Mild"
                elif cm_length < 25.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                    
            elif damage_class == "dent":
                # Dent depth calculation simulated using size of deformation
                pixel_diagonal = np.sqrt(w**2 + h**2)
                # Max depth estimation: roughly 5% of the diagonal dimension
                cm_depth = round(pixel_diagonal * cm_per_pixel * 0.05, 1)
                cm_depth = max(0.2, cm_depth) # Avoid 0.0 depth
                metrics["depth_cm"] = cm_depth
                metrics["width_cm"] = round(w * cm_per_pixel, 1)
                metrics["height_cm"] = round(h * cm_per_pixel, 1)
                
                # Classify severity
                if cm_depth < 0.8:
                    severity = "Mild"
                elif cm_depth < 2.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                    
            elif damage_class == "rust":
                # Rust is classified by spread area
                metrics["spread_area_cm2"] = cm2_area
                
                if cm2_area < 12.0:
                    severity = "Mild"
                elif cm2_area < 50.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                    
            elif damage_class in ["glass_shatter", "broken_lamp"]:
                metrics["spread_area_cm2"] = cm2_area
                
                if cm2_area < 20.0:
                    severity = "Mild"
                elif cm2_area < 75.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
            
            results.append({
                "id": det["id"],
                "class": damage_class,
                "confidence": det["confidence"],
                "box": det["box"],
                "severity": severity,
                "metrics": metrics
            })
            
        return results
