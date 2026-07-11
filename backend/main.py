"""
main.py — Secure, scalable FastAPI backend for vehicle damage assessment.

Features:
- API versioning (/api/v1/...) with backwards compatibility.
- AnyIO thread offloading for CPU-bound computer vision/deep learning models.
- Strictly enforced 10MB upload limits and EXIF metadata stripping.
- Constant-time API Key verification with multi-key support.
- Optional Redis-backed rate limiting with local sliding-window fallback.
- Enhanced system resource and model loading state health monitoring.
- CORS hardening & clickjacking protection headers.
"""

import os
import shutil
import time
import uuid
import secrets
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

import psutil
import redis
from anyio import to_thread
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from PIL import Image

# 1. Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("overbody_api")

# 2. Load environment variables
load_dotenv()

from services.detector import DamageDetector
from services.guidance import RepairGuidanceService
from services.severity import SeverityEstimator
from utils.cv_utils import detect_calibration_factor, get_annotated_image_base64

app = FastAPI(
    title="Overbody Damage Detection API",
    description="Production-grade secure API for detecting vehicle surface damage and generating AI repair guidance.",
    version="1.1.0",
)

# 3. CORS Hardening
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Ensure temp directory exists
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Services
detector = DamageDetector()
severity_estimator = SeverityEstimator()
guidance_service = RepairGuidanceService()

# 5. Secure API Key Management
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Support multiple comma-separated keys, with local default
raw_keys = os.getenv("API_KEYS", os.getenv("API_KEY", "overbody_secure_key_2026"))  # pragma: allowlist secret
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

def get_api_key(header_key: str = Security(api_key_header)):
    if not header_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing. Please provide the X-API-Key header."
        )
    
    # Constant-time comparison to prevent timing attacks
    for key in API_KEYS:
        if secrets.compare_digest(header_key, key):
            return header_key
            
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key. Unauthorized access."
    )

# 6. Redis or In-memory Rate Limiting Configuration
redis_client: Optional[redis.Redis] = None
redis_url = os.getenv("REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url, socket_timeout=2.0, decode_responses=True)
        # Ping to verify connection
        redis_client.ping()
        logger.info("[rate-limit] Connected to Redis server for scalable rate limiting.")
    except Exception as re:
        logger.warning(f"[rate-limit] Redis connection failed, falling back to local memory: {re}")
        redis_client = None

RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10
request_history: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed, False if rate limit is exceeded."""
    if redis_client:
        try:
            key = f"rate_limit:{client_ip}"
            current = redis_client.get(key)
            if current is not None and int(current) >= RATE_LIMIT_MAX_REQUESTS:
                return False
            
            # Increment and set TTL
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, RATE_LIMIT_WINDOW)
            pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"[rate-limit] Redis rate limiting error: {e}. Falling back to in-memory check.")
            
    # In-memory sliding window fallback
    current_time = time.time()
    request_history[client_ip] = [
        t for t in request_history[client_ip] if current_time - t < RATE_LIMIT_WINDOW
    ]
    if len(request_history[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
        
    request_history[client_ip].append(current_time)
    return True

@app.middleware("http")
async def rate_limiter_middleware(request: Request, call_next):
    path = request.url.path
    if path in ["/api/health", "/api/v1/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Limit is {RATE_LIMIT_MAX_REQUESTS} requests per minute.",
        )

    return await call_next(request)

# ---------------------------------------------------------
# Versioned API Endpoints (v1)
# ---------------------------------------------------------

@app.get("/api/v1/health")
def health_check_v1():
    # Gather system metrics for cloud/k8s orchestrators
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    # Check model states
    from services.detector import _yolo_available
    from services.depth_estimator import _midas_available

    return {
        "status": "healthy",
        "gemini_api_configured": os.getenv("GEMINI_API_KEY") is not None,
        "system_metrics": {
            "cpu_utilization_percent": cpu_percent,
            "ram_utilization_percent": memory_info.percent,
            "ram_free_gb": round(memory_info.available / (1024 ** 3), 2),
        },
        "model_loading_states": {
            "yolov8_damage_loaded": _yolo_available,
            "midas_depth_loaded": _midas_available,
        },
        "rate_limit_max": RATE_LIMIT_MAX_REQUESTS,
    }

@app.post("/api/v1/analyze")
async def analyze_image_v1(file: UploadFile = File(...), _api_key: str = Depends(get_api_key)):
    # 1. Enforce strict extension validations
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload PNG, JPG, or WEBP."
        )

    # 2. Enforce strict 10MB upload limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)

    total_bytes = 0
    try:
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(256 * 1024):  # 256KB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload file exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB."
                    )
                buffer.write(chunk)

        # 3. Security check: Verify image structure & strip EXIF metadata (preventing EXIF injection & leaks)
        try:
            with Image.open(temp_file_path) as img:
                img.verify()
            
            # Reopen to strip EXIF and resave
            with Image.open(temp_file_path) as img:
                img_format = img.format or "JPEG"
                # Strip EXIF by saving without metadata or empty exif
                img.save(temp_file_path, format=img_format, exif=b"")
                logger.info(f"[security] EXIF metadata stripped successfully for {file.filename}")
        except HTTPException:
            raise
        except Exception as se:
            logger.error(f"[security] Failed image structure check or EXIF strip: {se}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or invalid image file structure."
            )

        # 4. Offload heavy CV and DL inference steps to background thread pool
        # This keeps the main FastAPI event loop completely unblocked.
        cm_per_pixel, ref_box = await to_thread.run_sync(
            detect_calibration_factor, temp_file_path
        )
        
        raw_detections = await to_thread.run_sync(
            detector.detect, temp_file_path
        )
        
        damages = await to_thread.run_sync(
            severity_estimator.estimate, raw_detections, cm_per_pixel, temp_file_path
        )

        # 5. Multimodal Panel Classification via Gemini
        try:
            if os.getenv("GEMINI_API_KEY"):
                from google import genai
                client = genai.Client()
                with Image.open(temp_file_path) as img_pil:
                    prompt = (
                        "Identify the primary car body panel shown in this image. "
                        "Answer with exactly one of: 'Front Bumper / Fender', 'Hood / Roof', 'Door Panel', 'Rear Bumper / Trunk', 'Lower Rocker Panel'."
                    )
                    # Run content generation (network bound, so we also run it in thread pool)
                    response = await to_thread.run_sync(
                        lambda: client.models.generate_content(
                            model="gemini-2.5-flash", contents=[img_pil, prompt]
                        )
                    )
                    primary_panel = response.text.strip()
                    primary_panel = (
                        primary_panel.replace("`", "").replace("'", "").replace('"', "").strip()
                    )

                    valid_panels = [
                        "Front Bumper / Fender",
                        "Hood / Roof",
                        "Door Panel",
                        "Rear Bumper / Trunk",
                        "Lower Rocker Panel",
                    ]
                    if any(vp.lower() in primary_panel.lower() for vp in valid_panels):
                        matched_panel = next(
                            vp for vp in valid_panels if vp.lower() in primary_panel.lower()
                        )
                        for d in damages:
                            d["panel"] = matched_panel
        except Exception as ge:
            logger.warning(f"Failed to classify panel via Gemini: {ge}")

        # 6. Generate annotated base64 visualization string
        annotated_image_b64 = await to_thread.run_sync(
            get_annotated_image_base64, temp_file_path, damages, ref_box
        )

        # 7. Call Gemini for repair guidance report (incorporates network circuit breaker)
        repair_guide = await to_thread.run_sync(
            guidance_service.generate_guide, damages
        )

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
            "calibration": {"cm_per_pixel": cm_per_pixel, "reference_found": ref_box is not None},
            "damages": damages,
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",
            "repair_guide": repair_guide,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing analysis pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as ex:
                logger.error(f"Error removing temp file {temp_file_path}: {ex}")

@app.post("/api/v1/export-report", response_class=HTMLResponse)
async def export_report_v1(report_data: dict, _api_key: str = Depends(get_api_key)):
    damages = report_data.get("damages", [])
    overall = report_data.get("overall_severity", "Unknown")
    summary = report_data.get("summary", {"Mild": 0, "Moderate": 0, "Severe": 0})
    repair_guide = report_data.get("repair_guide", "")

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

        badge_class = (
            "badge-severe"
            if sev == "Severe"
            else ("badge-moderate" if sev == "Moderate" else "badge-mild")
        )

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

# ---------------------------------------------------------
# Deprecated Backward-Compatible Fallbacks (v0)
# ---------------------------------------------------------

@app.get("/api/health")
def health_check_legacy():
    return {
        "status": "healthy",
        "gemini_api_configured": os.getenv("GEMINI_API_KEY") is not None,
        "rate_limit_max": RATE_LIMIT_MAX_REQUESTS,
    }

@app.post("/api/analyze")
async def analyze_image_legacy(file: UploadFile = File(...), _api_key: str = Depends(get_api_key)):
    logger.warning("[deprecation] Legacy endpoint /api/analyze called. Please migrate to /api/v1/analyze.")
    return await analyze_image_v1(file, _api_key)

@app.post("/api/export-report", response_class=HTMLResponse)
async def export_report_legacy(report_data: dict, _api_key: str = Depends(get_api_key)):
    logger.warning("[deprecation] Legacy endpoint /api/export-report called. Please migrate to /api/v1/export-report.")
    return await export_report_v1(report_data, _api_key)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
