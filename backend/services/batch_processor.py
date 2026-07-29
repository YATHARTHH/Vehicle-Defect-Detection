import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger("overbody_api.batch_processor")


def process_batch(image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates individual image inspection results into a consolidated batch report.
    """
    total_images = len(image_results)
    total_defects = sum(r.get("total_defects", 0) for r in image_results)
    
    severity_counts = {"MILD": 0, "MODERATE": 0, "SEVERE": 0}
    all_findings = []

    for idx, r in enumerate(image_results):
        for d in r.get("defects", []):
            sev = d.get("severity", {}).get("rating", "MILD").upper()
            if sev in severity_counts:
                severity_counts[sev] += 1
            all_findings.append({
                "image_index": idx + 1,
                "filename": r.get("filename", f"Image_{idx+1}"),
                "panel": d.get("panel", "Unknown"),
                "class": d.get("class", "damage"),
                "confidence": d.get("confidence", 0.0),
                "severity": sev,
            })

    if severity_counts["SEVERE"] > 0:
        overall_batch_rating = "Severe Fleet Damage"
    elif severity_counts["MODERATE"] > 0:
        overall_batch_rating = "Moderate Fleet Damage"
    elif total_defects > 0:
        overall_batch_rating = "Mild Surface Scruff"
    else:
        overall_batch_rating = "Good Condition"

    return {
        "success": True,
        "batch_summary": {
            "total_images_processed": total_images,
            "total_defects_found": total_defects,
            "overall_batch_rating": overall_batch_rating,
            "severity_breakdown": severity_counts,
        },
        "itemized_results": image_results,
        "consolidated_findings": all_findings,
    }
