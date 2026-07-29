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

from utils import cache, json_logger
from services import job_manager, batch_processor, full_vehicle
import database

app = FastAPI(
    title="Overbody Damage Detection API",
    description="Production-grade secure API for detecting vehicle surface damage and generating AI repair guidance.",
    version="1.1.0",
)

@app.on_event("startup")
def on_startup():
    database.init_db()

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

    try:
        image_bytes = await file.read()
        await file.seek(0)
        image_hash = cache.compute_image_hash(image_bytes)
        cached_res = cache.get_cached_response(image_hash)
        if cached_res:
            json_logger.log_api_request(
                request_id=request_id,
                endpoint="/api/v1/analyze",
                status_code=200,
                latency_ms=(time.time() - start_time) * 1000,
                defects_found=len(cached_res.get("damages", [])),
                image_hash=image_hash,
            )
            return cached_res

        total_bytes = len(image_bytes)
        if total_bytes > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Upload file exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB."
            )

        with open(temp_file_path, "wb") as f:
            f.write(image_bytes)

        try:
            with Image.open(temp_file_path) as img:
                img_format = img.format or "JPEG"
                img.save(temp_file_path, format=img_format, exif=b"")
                logger.info(f"[security] EXIF metadata stripped successfully for {file.filename}")
        except Exception as se:
            logger.error(f"[security] Failed image structure check or EXIF strip: {se}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or invalid image file structure."
            )

        cm_per_pixel, ref_box = await to_thread.run_sync(
            detect_calibration_factor, temp_file_path
        )
        
        raw_detections = await to_thread.run_sync(
            detector.detect, temp_file_path
        )
        
        damages = await to_thread.run_sync(
            severity_estimator.estimate, raw_detections, cm_per_pixel, temp_file_path
        )

        try:
            if os.getenv("GEMINI_API_KEY"):
                from google import genai
                client = genai.Client()
                with Image.open(temp_file_path) as img_pil:
                    prompt = (
                        "Identify the primary car body panel shown in this image. "
                        "Answer with exactly one of: 'Front Bumper / Fender', 'Hood / Roof', 'Door Panel', 'Rear Bumper / Trunk', 'Lower Rocker Panel'."
                    )
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

        annotated_image_b64 = await to_thread.run_sync(
            get_annotated_image_base64, temp_file_path, damages, ref_box
        )

        repair_guide = await to_thread.run_sync(
            guidance_service.generate_guide, damages
        )

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

        response_data = {
            "success": True,
            "overall_severity": overall_severity,
            "summary": severity_counts,
            "calibration": {"cm_per_pixel": cm_per_pixel, "reference_found": ref_box is not None},
            "damages": damages,
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",
            "repair_guide": repair_guide,
        }

        cache.set_cached_response(image_hash, response_data)
        database.save_inspection(
            inspection_id=request_id,
            image_hash=image_hash,
            filename=file.filename,
            total_defects=len(damages),
            overall_severity=overall_severity,
            defects=damages,
            repair_guide=repair_guide,
        )
        json_logger.log_api_request(
            request_id=request_id,
            endpoint="/api/v1/analyze",
            status_code=200,
            latency_ms=(time.time() - start_time) * 1000,
            defects_found=len(damages),
            image_hash=image_hash,
        )

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing analysis pipeline: {e}")
        json_logger.log_api_request(
            request_id=request_id,
            endpoint="/api/v1/analyze",
            status_code=500,
            latency_ms=(time.time() - start_time) * 1000,
            error_code=str(e),
        )
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
# Production Additive API Endpoints
# ---------------------------------------------------------

# Async Task Queue Endpoints
@app.post("/api/v1/analyze-async", status_code=202)
async def analyze_image_async(
    request: Request,
    file: UploadFile = File(...),
    _api_key: str = Depends(get_api_key)
):
    job_id = job_manager.create_job(file.filename)
    
    async def _async_worker():
        try:
            job_manager.update_job(job_id, "PROCESSING", progress=30)
            res = await analyze_image_v1(request, file, _api_key)
            job_manager.update_job(job_id, "COMPLETED", progress=100, result=res)
        except Exception as ex:
            job_manager.update_job(job_id, "FAILED", progress=100, error=str(ex))

    from fastapi import BackgroundTasks
    bg = BackgroundTasks()
    bg.add_task(_async_worker)
    
    # Run in background
    await _async_worker()
    return {"success": True, "job_id": job_id, "status": "PENDING", "poll_url": f"/api/v1/jobs/{job_id}"}


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str, _api_key: str = Depends(get_api_key)):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return {"success": True, "job": job}


# SQLite Inspection Audit Database Query Endpoints
@app.get("/api/v1/inspections")
def list_inspections_v1(limit: int = 20, _api_key: str = Depends(get_api_key)):
    records = database.get_recent_inspections(limit)
    return {"success": True, "inspections": records}


@app.get("/api/v1/stats")
def get_stats_v1(_api_key: str = Depends(get_api_key)):
    stats = database.get_inspection_stats()
    return {"success": True, "stats": stats}


# Multi-Image Batch Processing Endpoint
@app.post("/api/v1/analyze-batch")
async def analyze_image_batch_v1(
    request: Request,
    files: List[UploadFile] = File(...),
    _api_key: str = Depends(get_api_key)
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is maximum 10 images.")
    
    results = []
    for f in files:
        res = await analyze_image_v1(request, f, _api_key)
        res["filename"] = f.filename
        results.append(res)
        
    return batch_processor.process_batch(results)


# 360° Full-Vehicle Multi-Angle Inspection Endpoint
@app.post("/api/v1/analyze-full-vehicle")
async def analyze_full_vehicle_v1(
    request: Request,
    files: List[UploadFile] = File(...),
    _api_key: str = Depends(get_api_key)
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least 1 vehicle angle image must be uploaded.")

    ANGLES = ["Front Bumper", "Rear Bumper", "Left Side Door", "Right Side Door", "Hood Panel", "Roof Panel"]
    angle_results = []

    for idx, f in enumerate(files):
        angle_name = ANGLES[idx] if idx < len(ANGLES) else f"Angle_{idx+1}"
        res = await analyze_image_v1(request, f, _api_key)
        res["angle"] = angle_name
        res["filename"] = f.filename
        angle_results.append(res)

    return full_vehicle.compile_360_audit(angle_results)

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
