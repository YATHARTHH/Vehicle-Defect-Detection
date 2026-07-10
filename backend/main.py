import os
import shutil
import uuid
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

from services.detector import DamageDetector
from utils.cv_utils import detect_calibration_factor, get_annotated_image_base64
from services.severity import SeverityEstimator
from services.guidance import RepairGuidanceService

app = FastAPI(
    title="Overbody Damage Detection API",
    description="Secure API for detecting vehicle surface damage, estimating severity, and generating AI repair guidance.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Services
detector = DamageDetector()
severity_estimator = SeverityEstimator()
guidance_service = RepairGuidanceService()

# ---------------------------------------------------------
# Security & Rate Limiting Implementations
# ---------------------------------------------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(header_key: str = Security(api_key_header)):
    """
    Dependency to validate X-API-Key header.
    """
    configured_key = os.getenv("API_KEY", "overbody_secure_key_2026")
    if not header_key:
        raise HTTPException(
            status_code=401,
            detail="API Key missing. Please provide the X-API-Key header."
        )
    if header_key != configured_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Unauthorized access."
        )
    return header_key

# Simple in-memory sliding window rate limiter
RATE_LIMIT_WINDOW = 60 # seconds
RATE_LIMIT_MAX_REQUESTS = 10 # limit requests per IP
request_history: Dict[str, List[float]] = defaultdict(list)

@app.middleware("http")
async def rate_limiter_middleware(request: Request, call_next):
    """
    Middleware to rate limit requests per client IP.
    """
    # Exclude health check and documentation endpoints from rate limit
    path = request.url.path
    if path in ["/api/health", "/docs", "/openapi.json"]:
        return await call_next(request)
        
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Filter timestamps to keep only those within the sliding window
    request_history[client_ip] = [
        t for t in request_history[client_ip]
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    if len(request_history[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {RATE_LIMIT_MAX_REQUESTS} requests per minute."
        )
        
    request_history[client_ip].append(current_time)
    response = await call_next(request)
    return response

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "gemini_api_configured": os.getenv("GEMINI_API_KEY") is not None,
        "rate_limit_max": RATE_LIMIT_MAX_REQUESTS
    }

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    _api_key: str = Depends(get_api_key)
):
    # Validate file format by extension first
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PNG, JPG, or WEBP.")

    # Save file to a temporary unique location
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Security check: verify image integrity using Pillow to prevent exploit uploads
        try:
            with Image.open(temp_file_path) as img:
                img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Corrupted or invalid image file structure.")

        # Reopen image as verify() closes the file handles
        # 1. Detect pixel-to-cm calibration factor
        cm_per_pixel, ref_box = detect_calibration_factor(temp_file_path)
        
        # 2. Run damage detector to locate boxes and contours
        raw_detections = detector.detect(temp_file_path)
        
        # 3. Estimate physical measurements and severity classes
        damages = severity_estimator.estimate(raw_detections, cm_per_pixel)
        
        # 4. Generate annotated image base64 string
        annotated_image_b64 = get_annotated_image_base64(temp_file_path, damages, ref_box)
        
        # 5. Call Gemini to get repair guidance report
        repair_guide = guidance_service.generate_guide(damages)
        
        # Calculate summary statistics
        severity_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for d in damages:
            severity_counts[d["severity"]] += 1
            
        overall_severity = "Good"
        if severity_counts["Severe"] > 0:
            overall_severity = "Severe"
        elif severity_counts["Moderate"] > 0:
            overall_severity = "Moderate"
        elif severity_counts["Mild"] > 0:
            overall_severity = "Mild"

        return {
            "success": True,
            "overall_severity": overall_severity,
            "summary": severity_counts,
            "calibration": {
                "cm_per_pixel": cm_per_pixel,
                "reference_found": ref_box is not None
            },
            "damages": damages,
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",
            "repair_guide": repair_guide
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error executing analysis pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as ex:
                print(f"Error removing temp file {temp_file_path}: {ex}")

@app.post("/api/export-report", response_class=HTMLResponse)
async def export_report(report_data: dict, _api_key: str = Depends(get_api_key)):
    """
    Generates a print-ready, professional HTML assessment report page.
    """
    damages = report_data.get("damages", [])
    overall = report_data.get("overall_severity", "Unknown")
    summary = report_data.get("summary", {"Mild": 0, "Moderate": 0, "Severe": 0})
    repair_guide = report_data.get("repair_guide", "")
    
    # Convert guide markdown simple lists and headers to HTML for formatting
    guide_html = repair_guide.replace("\n", "<br/>")
    
    damages_rows = ""
    for d in damages:
        cls = d["class"].replace("_", " ").title()
        sev = d["severity"]
        conf = int(d["confidence"] * 100)
        metrics = d["metrics"]
        size_desc = f"Area: {metrics.get('cm2_area')} cm²"
        if "length_cm" in metrics:
            size_desc += f", Length: {metrics['length_cm']} cm"
        if "depth_cm" in metrics:
            size_desc += f", Depth: {metrics['depth_cm']} cm"
            
        badge_class = "badge-severe" if sev == "Severe" else ("badge-moderate" if sev == "Moderate" else "badge-mild")
        
        damages_rows += f"""
        <tr>
            <td><strong>{cls}</strong></td>
            <td><span class="badge {badge_class}">{sev}</span></td>
            <td>{conf}%</td>
            <td>{size_desc}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Inspection Assessment Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; margin: 40px; line-height: 1.6; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
            .title {{ font-size: 28px; font-weight: bold; text-transform: uppercase; }}
            .metadata {{ text-align: right; font-size: 14px; color: #666; }}
            .status-box {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; display: flex; justify-content: space-between; }}
            .status-value {{ font-size: 20px; font-weight: bold; }}
            .severe {{ color: #dc3545; }}
            .moderate {{ color: #ffc107; }}
            .mild {{ color: #28a745; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; text-transform: uppercase; }}
            .badge-severe {{ background-color: #dc3545; }}
            .badge-moderate {{ background-color: #fd7e14; }}
            .badge-mild {{ background-color: #28a745; }}
            .guide-section {{ background-color: #fafbfc; border-left: 4px solid #0056b3; padding: 20px; border-radius: 0 8px 8px 0; margin-top: 30px; }}
            .print-btn {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 14px; font-weight: bold; cursor: pointer; float: right; margin-top: 20px; }}
            @media print {{
                .print-btn {{ display: none; }}
                body {{ margin: 20px; }}
            }}
        </style>
    </head>
    <body>
        <button class="print-btn" onclick="window.print()">Print / Save PDF</button>
        <div class="header">
            <div>
                <div class="title">Vehicle Condition Report</div>
                <div style="font-size: 14px; color: #555;">Automated Overbody Defect Assessment</div>
            </div>
            <div class="metadata">
                <div><strong>Date:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div><strong>Ref ID:</strong> {uuid.uuid4().hex[:8].upper()}</div>
            </div>
        </div>

        <div class="status-box">
            <div>
                <span style="color: #666; font-size: 14px; display: block;">Overall Severity Rating</span>
                <span class="status-value {overall.lower()}">{overall.upper()} DAMAGE DETECTED</span>
            </div>
            <div style="text-align: right;">
                <span style="color: #666; font-size: 14px; display: block;">Finding Summary</span>
                <strong>{summary.get('Mild', 0)} Mild | {summary.get('Moderate', 0)} Moderate | {summary.get('Severe', 0)} Severe</strong>
            </div>
        </div>

        <h3>Damage Location Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Damage Type</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>Physical Bounding Size</th>
                </tr>
            </thead>
            <tbody>
                {damages_rows}
            </tbody>
        </table>

        <div class="guide-section">
            <h3>AI Repair Guidance Recommendations</h3>
            <div style="font-size: 14px; line-height: 1.8;">
                {guide_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
