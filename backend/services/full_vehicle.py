import logging
from typing import List, Dict, Any

logger = logging.getLogger("overbody_api.full_vehicle")


def compile_360_audit(angle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compiles multi-angle (Front, Rear, Left, Right, Hood, Roof) vehicle inspections
    into a unified 360° audit report with a Composite Vehicle Health Score (0-100).
    """
    total_angles_submitted = len(angle_results)
    REQUIRED_ANGLES = ["Front", "Rear", "Left Side", "Right Side"]
    submitted_angles = [r.get("angle", "Unknown") for r in angle_results]

    # Coverage Completeness Index %
    matched_required = sum(1 for a in REQUIRED_ANGLES if any(a.lower() in s.lower() for s in submitted_angles))
    coverage_index = round((matched_required / float(len(REQUIRED_ANGLES))) * 100, 1)

    total_defects_across_vehicle = 0
    total_penalty = 0
    angle_summaries = []

    for r in angle_results:
        defects = r.get("defects", [])
        num_defects = len(defects)
        total_defects_across_vehicle += num_defects

        for d in defects:
            sev = d.get("severity", {}).get("rating", "MILD").upper()
            if sev == "SEVERE":
                total_penalty += 25
            elif sev == "MODERATE":
                total_penalty += 12
            else:
                total_penalty += 5

        angle_summaries.append({
            "angle": r.get("angle", "Custom View"),
            "filename": r.get("filename", "image.jpg"),
            "defects_count": num_defects,
            "overall_severity": r.get("overall_severity", "Good"),
            "defects": defects,
        })

    # Health Score Calculation (Starts at 100, max deduction 100)
    health_score = max(0, 100 - total_penalty)

    # Health Grade
    if health_score >= 90:
        health_grade = "Grade A (Excellent)"
    elif health_score >= 75:
        health_grade = "Grade B (Minor Wear)"
    elif health_score >= 60:
        health_grade = "Grade C (Moderate Damage)"
    elif health_score >= 40:
        health_grade = "Grade D (Major Repair Required)"
    else:
        health_grade = "Grade F (Severe Body Collision)"

    return {
        "success": True,
        "full_vehicle_summary": {
            "vehicle_health_score": health_score,
            "vehicle_health_grade": health_grade,
            "coverage_completeness_index": f"{coverage_index}%",
            "total_angles_inspected": total_angles_submitted,
            "total_defects_detected": total_defects_across_vehicle,
        },
        "angle_breakdown": angle_summaries,
    }
