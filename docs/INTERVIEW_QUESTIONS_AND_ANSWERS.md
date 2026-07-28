# 🎯 Overbody Damage Detection — Comprehensive Technical Interview Guide & Q&A Handbook

> **Overview:** This document is an easy-to-understand, deep-dive interview preparation guide for the **Overbody Vehicle Damage Detection** project. It breaks down complex engineering concepts into simple real-world analogies, step-by-step technical explanations, code walkthroughs, and system design insights for software engineers, AI/ML developers, and system architects.

---

## 📚 Table of Contents
1. [System Architecture & Async Backend Processing](#1-system-architecture--async-backend-processing)
2. [Computer Vision, YOLOv8 & Classical CV Fallbacks](#2-computer-vision-yolov8--classical-cv-fallbacks)
3. [Monocular 3D Depth & Real-World Physical Calibration](#3-monocular-3d-depth--real-world-physical-calibration)
4. [Generative AI, Multimodal LLMs & Circuit Breaker Resilience](#4-generative-ai-multimodal-llms--circuit-breaker-resilience)
5. [Security Engineering & OWASP Compliance](#5-security-engineering--owasp-compliance)
6. [Frontend Engineering & Interactive Visualization](#6-frontend-engineering--interactive-visualization)
7. [System Design, Scalability & MLOps Scenarios](#7-system-design-scalability--mlops-scenarios)

---

## 1. System Architecture & Async Backend Processing

### Q1: Why was FastAPI chosen for the backend over frameworks like Flask or Django, and how does it run heavy AI models without freezing the server?

#### 💡 Simple Analogy
Imagine a busy restaurant waiter (the server).
- In **Flask (Synchronous)**, when a customer orders food that takes 10 minutes to cook (AI model inference), the waiter stands in the kitchen waiting for 10 minutes. No other customer can place an order or get served until that dish is done!
- In **FastAPI (Asynchronous)**, the waiter takes the order, passes it to a kitchen helper thread, and instantly walks back to take orders from other customers. Once the dish is ready, the waiter brings it back to the first customer.

#### 🛠️ Easy & Detailed Technical Explanation
1. **The Asynchronous Event Loop:** FastAPI runs on an ASGI server (`Uvicorn`). It uses a single thread called an **Event Loop** to handle thousands of network connections efficiently.
2. **The CPU-Blocking Problem:** AI inference tasks (like YOLOv8 bounding box detection and Intel MiDaS 3D depth computation) require intense mathematical processing on PyTorch and OpenCV. If executed directly inside an `async def` route function, these heavy CPU operations will block the event loop, causing all incoming requests to freeze.
3. **The Solution — Thread Pool Offloading:** We use `anyio.to_thread.run_sync()` to offload heavy blocking functions to background worker threads:
   ```python
   # The main event loop delegates the heavy work to a worker thread:
   raw_detections = await to_thread.run_sync(detector.detect, temp_file_path)
   ```
   - **Why this works:** The main event loop remains 100% free to accept incoming HTTP requests while worker threads perform PyTorch/OpenCV calculations in parallel.

---

### Q2: Walk me through the step-by-step pipeline when a user uploads a vehicle photo until the final report is generated.

#### 🛠️ Step-by-Step Data Flow Breakdown

```
[ 📤 User Uploads Image ]
          │
          ▼
1. 🛡️ Security Check (main.py)
   ├── Validate extension (.jpg, .png, .webp)
   ├── Stream file in 256KB chunks to enforce 10MB limit (prevents RAM crash)
   └── Strip EXIF metadata with Pillow (removes GPS & camera IDs for privacy)
          │
          ▼
2. 📐 Physical Scale Calibration (cv_utils.py)
   ├── Search image for standard green credit card reference box via HSV color masking
   └── Calculate pixels-per-cm ratio (cm_per_pixel). Fallback = 0.04 cm/px if no card found
          │
          ▼
3. 🔍 Damage Detection Engine (detector.py)
   ├── Primary Path: Run fine-tuned YOLOv8 (abdullahg7/cardd-yolov8s) to find boxes & labels
   └── Fallback Path: If YOLO fails, run OpenCV Sobel/Canny/HSV classical CV routines
          │
          ▼
4. 📏 Depth & Severity Engine (severity.py)
   ├── Run Intel MiDaS depth model to estimate dent depth in centimeters (cm_depth)
   ├── Calculate scratch length (cm) and rust spread area (cm²) using calibration factor
   └── Categorize defects into "Mild", "Moderate", or "Severe"
          │
          ▼
5. 🤖 Panel Classification & Repair Guide (guidance.py & main.py)
   ├── Call Gemini 2.5 Flash Vision to identify panel name (e.g., "Front Bumper / Fender")
   └── Call Gemini 2.5 Flash Text to generate repair guide (or rule-based generator if circuit breaker trips)
          │
          ▼
6. 🖼️ Image Annotation & Cleanup (cv_utils.py)
   ├── Draw color-coded bounding boxes and physical measurements onto image using OpenCV
   ├── Encode annotated image as Base64 JPEG string
   └── Delete temp upload file in a `finally:` block to prevent disk space leaks
          │
          ▼
[ 📦 Return JSON Response to React Dashboard ]
```

---

### Q3: Why load AI models (YOLOv8 & MiDaS) lazily on server startup rather than loading them inside the request handler?

#### 💡 Simple Analogy
Imagine buying a heavy kitchen blender every time you want to make a smoothie, unpacking it, blending the fruit, and throwing the blender in the trash. That would be absurdly slow and wasteful! Instead, you keep the blender on the kitchen counter permanently so it's ready instantly.

#### 🛠️ Technical Detail
- **Heavy Model Overhead:** Loading PyTorch weights into memory takes **2 to 5 seconds** and allocates several hundred megabytes of RAM.
- **Lazy Singleton Cache Pattern:**
  ```python
  _yolo_model = None
  _yolo_available = None  # None = not tried yet

  def _try_load_yolo():
      global _yolo_model, _yolo_available
      if _yolo_available is not None:
          return _yolo_available  # Return cached result instantly!

      try:
          model_path = hf_hub_download("abdullahg7/cardd-yolov8s", "v2.0/best.pt")
          _yolo_model = YOLO(model_path)
          _yolo_available = True
      except Exception:
          _yolo_available = False
      return _yolo_available
  ```
- **Benefits:**
  1. **Zero-Latency Inferences:** Model weights load **once** during the first request (or server boot) and remain in memory.
  2. **Fast Response Times:** Sub-second inference per request instead of adding 5+ seconds of startup overhead every time.

---

### Q4: How does the server handle temporary uploaded files safely without risking memory leaks or disk space exhaustion?

#### 🛠️ Technical Detail
1. **Unique Filenames:** Every uploaded file is assigned a random UUID (e.g., `temp_uploads/a1b2c3d4-photo.jpg`). This avoids name collisions when multiple users upload files simultaneously.
2. **Chunked Streaming:** Rather than reading an entire 10MB file into RAM at once (`await file.read()`), we stream it in **256KB chunks**:
   ```python
   total_bytes = 0
   with open(temp_file_path, "wb") as buffer:
       while chunk := await file.read(256 * 1024):
           total_bytes += len(chunk)
           if total_bytes > 10 * 1024 * 1024: # 10MB Limit
               raise HTTPException(413, "File exceeds 10MB limit")
           buffer.write(chunk)
   ```
3. **Guaranteed File Cleanup with `finally`:**
   ```python
   try:
       # Process image pipeline...
   finally:
       if os.path.exists(temp_file_path):
           os.remove(temp_file_path)  # Always executes, even if error occurs!
   ```
   Using a `try...finally` block guarantees temporary files are deleted immediately after processing, preventing disk bloat.

---

## 2. Computer Vision, YOLOv8 & Classical CV Fallbacks

### Q5: Why use YOLOv8 for damage detection instead of older architectures like Faster R-CNN?

#### 💡 Simple Analogy
- **Faster R-CNN (Two-Stage Detector):** Like inspecting a room in two steps — first, draw candidate boxes around every object, then examine each box one by one to determine what it is. It's thorough but slow (~300ms per frame).
- **YOLOv8 (Single-Stage Detector):** Like taking a single glance at the room and immediately spotting and naming all objects simultaneously. It's blazingly fast (~30ms per frame).

#### 🛠️ Technical Comparison Table

| Metric / Feature | YOLOv8s (Used in Project) | Faster R-CNN | Mask R-CNN |
| :--- | :--- | :--- | :--- |
| **Architecture** | Single-Stage Regression | Two-Stage (RPN + Classifier) | Two-Stage + Mask Branch |
| **Inference Speed** | **Fast** (~30–50 ms on GPU) | **Slow** (~200–400 ms) | **Very Slow** (~400–700 ms) |
| **Model Size** | Lightweight (~22 MB) | Heavy (~150 MB) | Very Heavy (~250 MB) |
| **Domain Adaptation** | Pre-trained on CarDD dataset | Generic COCO pre-training | Generic COCO pre-training |
| **Output** | Bounding boxes + Labels | Bounding boxes + Labels | Polygon Masks + Boxes |

---

### Q6: Explain how the Classical Computer Vision fallback engine works without neural networks when YOLOv8 is offline.

#### 💡 Simple Analogy
If an expert AI brain goes offline, we use standard mathematical image filters (like photographic lenses) to look for known physical signatures: rust color, sharp edges of scratches, and lighting gradients of dents.

#### 🛠️ Detailed Breakdown of the 4 Classical Algorithms

1. **Rust Detection via HSV Color Space Masking:**
   - RGB color space is heavily affected by lighting/shadows. We convert images to **HSV (Hue, Saturation, Value)** because Hue isolates pure color.
   - Rust has a distinct orange-red hue ($0^\circ - 18^\circ$ and $160^\circ - 180^\circ$). We filter pixels matching this range to create a binary mask.
2. **Dent Detection via Sobel Filter Gradients:**
   - Dents deform smooth metal surfaces, causing light to fall off quickly across edges, creating shadow gradients.
   - We calculate 2D spatial image intensity gradients using 5x5 Sobel kernels:
     $$G_x = \text{Sobel}_x(\text{image}), \quad G_y = \text{Sobel}_y(\text{image}), \quad G = \sqrt{G_x^2 + G_y^2}$$
   - Regions with high gradient magnitude ($G$) indicate sudden light drops caused by surface indentations.
3. **Scratch Detection via Canny Edge Detection:**
   - Scratches create thin, sharp high-contrast line transitions where paint is cut.
   - Canny edge detection finds sharp intensity shifts and suppresses weak non-edge pixels to isolate crisp line structures.
4. **Geometric Shape Classification:**
   Once contours (shapes) are detected, we classify them using mathematical shape metrics:
   $$\text{Circularity} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$$
   - **Dent:** High circularity ($> 0.25$) — roughly round shape.
   - **Scratch:** Low aspect ratio / high elongation ($> 4.5$) — long thin strip.
   - **Crack:** Low circularity ($< 0.12$) — jagged, irregular branching shape.

---

### Q7: How does the system automatically identify which vehicle panel is damaged when Gemini AI is offline?

#### 🛠️ Spatial Zone Heuristics
When offline, the backend evaluates the **center coordinates** $(cx, cy)$ of the damage bounding box relative to total image width ($iw$) and height ($ih$):

```
┌─────────────────────────────────────────────────────────┐ 0% height
│                 Hood / Roof  (cy < 35%)                 │
├───────────────────┬──────────────────┬──────────────────┤ 35% height
│   Front Bumper    │    Door Panel    │   Rear Bumper    │
│    (cx < 25%)     │  (25% <= cx <= 75%)│    (cx > 75%)   │
├───────────────────┴──────────────────┴──────────────────┤ 75% height
│            Lower Rocker Panel  (cy > 75%)               │
└─────────────────────────────────────────────────────────┘ 100% height
```

```python
def _classify_panel(x, y, bw, bh, iw, ih) -> str:
    cx = x + (bw / 2)  # X-center of box
    cy = y + (bh / 2)  # Y-center of box

    if cy < ih * 0.35: return "Hood / Roof"
    if cy > ih * 0.75: return "Lower Rocker Panel"
    if cx < iw * 0.25: return "Front Bumper / Fender"
    if cx > iw * 0.75: return "Rear Bumper / Trunk"
    return "Door Panel"
```

---

### Q8: How does OpenCV render visual annotations on the image and send it to the frontend?

#### 🛠️ Technical Steps
1. **Draw Bounding Boxes:** `cv2.rectangle()` draws color-coded rectangles around detected defects:
   - 🟢 **Green:** Mild severity
   - 🟡 **Yellow:** Moderate severity
   - 🔴 **Red:** Severe defect
2. **Overlay Text Labels:** `cv2.putText()` renders text tags above each box displaying the class name and physical metric (e.g., `"dent | 1.8 cm depth"`).
3. **Base64 Encoding:** Instead of saving the annotated image to disk and serving a public URL, we convert the image array directly into memory buffer bytes and format it as a Base64 string:
   ```python
   _, buffer = cv2.imencode(".jpg", annotated_img)
   base64_str = base64.b64encode(buffer).decode("utf-8")
   ```
4. **Benefit:** Frontends can embed Base64 strings directly in HTML image tags `<img src="data:image/jpeg;base64,..." />` without requiring extra image fetch requests!

---

## 3. Monocular 3D Depth & Real-World Physical Calibration

### Q9: How do you convert 2D image pixels into real-world physical centimeters without depth cameras?

#### 💡 Simple Analogy
If you take a picture of a coin next to a unknown object, you can tell how big the object is because you already know the exact real-world size of a coin.

#### 🛠️ Technical Math Step-by-Step
1. **Known Standard:** According to the international **ISO/IEC 7810 standard**, every standard credit card in the world is exactly **8.56 cm long** by **5.398 cm wide**.
2. **OpenCV Calibration Steps:**
   - Convert image to HSV space to isolate green reference calibration cards/markers.
   - Find contours with `cv2.findContours()` and simplify them to 4 corners using `cv2.approxPolyDP()`.
   - Extract bounding box pixel width ($w_{\text{px}}$) and height ($h_{\text{px}}$).
3. **Scale Factor Calculation:**
   $$\text{pixels\_per\_cm} = \frac{\max(w_{\text{px}}, h_{\text{px}})}{8.56 \text{ cm}}$$
   $$\text{cm\_per\_pixel} = \frac{1.0}{\text{pixels\_per\_cm}}$$
4. **Measuring Flaws:** To find the physical length of a scratch with pixel length $L_{\text{px}}$:
   $$\text{Length}_{\text{cm}} = L_{\text{px}} \times \text{cm\_per\_pixel}$$

---

### Q10: How does Intel MiDaS calculate 3D dent depth from a standard 2D flat photo?

#### 💡 Simple Analogy
Human eyes can tell how deep a dent in a car door is from a 2D photo because our brain analyzes subtle shadows, surface curves, and light reflections. Intel MiDaS is a neural network trained on millions of 3D stereo scenes to replicate this exact human visual perception.

#### 🛠️ Technical Depth Formula Explained
1. **Inverse Depth Map:** MiDaS processes the image crop and produces a continuous relative depth map where pixel values represent inverse relative distance from the camera lens.
2. **Depth Calculation Code (`severity.py`):**
   ```python
   # Extract depth map region for the dent bounding box:
   crop_depth = depth_map[y:y+h, x:x+w]

   # 1. Calculate relative depth range between 95th and 5th percentiles (removes noise):
   depth_range = np.percentile(crop_depth, 95) - np.percentile(crop_depth, 5)

   # 2. Get real-world size of the dent in centimeters:
   max_dim_cm = max(w, h) * cm_per_pixel

   # 3. Calculate physical depth in cm using empirical scaling constant (0.12):
   depth_cm = depth_range * max_dim_cm * 0.12
   ```
3. **Why 95th and 5th Percentiles?** Using percentiles instead of absolute `max() - min()` prevents single noisy outlier pixels from ruining depth accuracy.

---

### Q11: What are the physical limitations of credit card calibration and monocular depth estimation?

#### 🛠️ Engineering Limitations & Mitigations

1. **Perspective Tilt Distortion:**
   - *Problem:* If the camera is held at an angle ($45^\circ$) rather than perpendicular ($90^\circ$), objects farther from the lens appear smaller in pixels.
   - *Mitigation:* The system uses `max(width, height)` of the detected card contour to minimize scale errors caused by mild tilt.
2. **Depth Scale Ambiguity:**
   - *Problem:* Monocular depth networks infer *relative* depth, not absolute distance metrics.
   - *Mitigation:* We tie relative depth variation directly to real-world bounding box dimensions ($\text{max\_dim\_cm}$) scaled by an empirical deformation constant (`0.12`) derived from car body sheet metal curvature benchmarks.

---

## 4. Generative AI, Multimodal LLMs & Circuit Breaker Resilience

### Q12: What is the Circuit Breaker pattern, and why was it built for Gemini AI?

#### 💡 Simple Analogy
Think of the electrical circuit breaker (fuse box) in your house. If an appliance shorts out, the fuse trips to cut off electricity and protect your home from catching fire.
In software, if an external API (like Gemini AI) starts failing or timing out, a **Circuit Breaker** stops making network calls to it. This prevents server lag and instantly switches to an offline fallback generator.

#### 🛠️ The 3 States of the Circuit Breaker

```
   +-------------------------------------------------------+
   |                                                       |
   |   +------------+  3 consecutive failures  +---------+ |
   +-->|   CLOSED   |------------------------->| TRIPPED |-+
       |  (Normal)  |                          | (Fails) |
       +------------+                          +---------+
             ▲                                      │
             │                                 300s timeout
             │                                      │
             │         +-----------+                │
             +---------| HALF-OPEN |<---------------+
               Success |  (Probe)  |
                       +-----------+
```

1. **`CLOSED` State (Normal):** All API calls go to Gemini AI. Consecutive failures are tracked.
2. **`TRIPPED` State (Protection Mode):** If Gemini fails 3 times in a row, the breaker trips for **300 seconds (5 minutes)**. During these 5 minutes:
   - Zero network requests are sent to Gemini.
   - The backend instantly generates structured repair reports using a deterministic local Python template engine.
3. **`HALF-OPEN` State (Recovery Testing):** After 5 minutes, the breaker allows **1 test request** through to check if Gemini recovered. If successful, it resets to `CLOSED`. If it fails, it re-trips for another 5 minutes.

---

### Q13: How does Gemini 2.5 Flash perform multimodal panel identification and repair guidance?

#### 🛠️ Technical Details
- **Multimodal Vision Prompting:** We supply the raw image binary along with a prompt instructing Gemini to analyze visual geometry and identify the exact vehicle panel name (e.g., *"Hood"*, *"Front Left Fender"*, *"Rear Bumper"*).
- **Structured Text Prompting:** We supply damage metadata (defect types, severity ratings, physical measurements) to Gemini text completion to generate structured Markdown repair instructions:
  - Required professional tools & materials (e.g., body filler, dual-action sander, primer).
  - Step-by-step repair workflow.
  - Estimated DIY cost vs. Professional Body Shop repair cost.

---

## 5. Security Engineering & OWASP Compliance

### Q14: How does the backend prevent timing attack exploits during API key validation?

#### 💡 Simple Analogy
Imagine trying to guess a secret password on a digital lock.
- **Normal String Comparison (`==`):** If you guess `"AXXXX"`, the lock takes 1 millisecond to reject it because the first letter is wrong. But if you guess `"BXXXX"` and the first letter is correct, the lock takes 2 milliseconds to check the second letter before rejecting it! By using a stopwatch, an attacker can figure out the password character by character.
- **Constant-Time Comparison (`secrets.compare_digest`):** The lock ALWAYS takes exactly 2 milliseconds to reply, regardless of how many characters are correct or incorrect. The attacker learns nothing from the timing!

#### 🛠️ Code Implementation
```python
def get_api_key(header_key: str = Security(api_key_header)):
    for key in API_KEYS:
        # Constant-time comparison prevents timing side-channel attacks:
        if secrets.compare_digest(header_key, key):
            return header_key
    raise HTTPException(status_code=401, detail="Invalid API Key.")
```

---

### Q15: Why is EXIF metadata stripping implemented, and how does it protect user privacy?

#### 🛠️ Technical Details
- **The Privacy Vulnerability:** Smartphone cameras embed hidden EXIF headers in JPEG images containing sensitive metadata:
  - Precise GPS coordinates (latitude/longitude of user's home/accident site).
  - Exact date and timestamp.
  - Camera device model and serial number.
- **Security Stripping with Pillow (`main.py`):**
  ```python
  with Image.open(temp_file_path) as img:
      img.verify()  # Confirm it's a valid image structure

  with Image.open(temp_file_path) as img:
      # Save image back to disk with empty EXIF metadata byte string:
      img.save(temp_file_path, format=img_format, exif=b"")
  ```
  This guarantees sanitized images returned to inspectors or stored in databases contain zero location or device identifiers.

---

### Q16: How does rate limiting protect the API using Redis sliding windows with an in-memory fallback?

#### 🛠️ Architecture Design
- **Redis Path (Production Multi-Server Setup):**
  Uses Redis atomic pipelines (`INCR` and `EXPIRE`) to track IP request counts in a shared central database across multiple API server nodes.
- **In-Memory Path (Development / Single-Server Setup):**
  Maintains an in-memory dictionary storing timestamps of client IP requests, discarding timestamps older than the sliding window interval (e.g., 60 seconds).
- **Enforcement:** If an IP exceeds 60 requests per minute, the server responds with `429 Too Many Requests`.

---

### Q17: What OWASP Security Headers are injected into HTTP responses and why?

#### 🛠️ Security Middleware Code (`main.py`)
```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"            # 1. Anti-Clickjacking
    response.headers["X-Content-Type-Options"] = "nosniff"  # 2. Anti-MIME Sniffing
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response
```

1. **`X-Frame-Options: DENY`:** Prevents malicious websites from embedding our web dashboard inside an invisible `<iframe>` to trick users into performing hidden actions (Clickjacking).
2. **`X-Content-Type-Options: nosniff`:** Stops browsers from trying to guess file types (MIME-sniffing), preventing malicious file uploads from executing as JavaScript.

---

## 6. Frontend Engineering & Interactive Visualization

### Q18: How does the interactive SVG ruler calculate real-world distance between user clicks on screen?

#### 🛠️ Coordinate Mapping Step-by-Step

```
[ User Clicks Screen Point (clickX, clickY) ]
                    │
                    ▼
Step 1: Translate Screen Coordinates to Original Natural Image Pixels
 naturalX = ((clickX - rect.left) / rect.width)  * naturalWidth
 naturalY = ((clickY - rect.top)  / rect.height) * naturalHeight
                    │
                    ▼
Step 2: Calculate Euclidean Distance in Original Pixel Space
 dx = endPoint.naturalX - startPoint.naturalX
 dy = endPoint.naturalY - startPoint.naturalY
 distance_px = sqrt(dx^2 + dy^2)
                    │
                    ▼
Step 3: Convert Pixel Distance to Physical Centimeters
 distance_cm = distance_px * cm_per_pixel
```

- **Why store coordinates in Natural Image Space?**
  If user clicks were stored in screen display pixels, resizing the browser window would mess up the ruler line position. Storing points relative to original image dimensions keeps ruler lines accurate across all screen sizes!

---

### Q19: How is the split-screen image comparison slider implemented for 60 FPS performance without lag?

#### 💡 Simple Analogy
Instead of having JavaScript constantly redraw pixels on a canvas whenever you drag the slider, we stack two images on top of each other and use a CSS "scissors" effect (`clip-path`) controlled by the computer's GPU.

#### 🛠️ Technical Details
- **React Component Structure:**
  ```tsx
  <div style={{ "--split-x": `${splitPos}%` } as React.CSSProperties}>
    <img className="split-base" src={originalImage} />
    <img className="split-overlay" src={annotatedImageBase64} />
  </div>
  ```
- **CSS GPU Clipping:**
  ```css
  .split-overlay {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    clip-path: inset(0 0 0 var(--split-x)); /* Crop left edge based on slider variable */
  }
  ```
- **Performance Advantage:** Updating `--split-x` triggers hardware-accelerated GPU compositing instead of CPU repaints, keeping animation at a buttery-smooth **60 frames per second**.

---

### Q20: Why choose simple React state (`useState`/`useRef`) over Redux or Zustand for this dashboard?

#### 🛠️ Architectural Rationale
- **Avoid Over-Engineering:** The application dashboard is a focused single-view inspection workbench. State (uploaded file, API response data, ruler start/end points, slider position) is local to `App.tsx`.
- **No Deep Prop Drilling:** Component trees are compact, eliminating the need for global store boilerplate like Redux slices, actions, and reducers.
- **Zero Bundle Bloat:** Keeping state native to React keeps bundle sizes small and initial page loads super fast.

---

## 7. System Design, Scalability & MLOps Scenarios

### Q21: How would you scale this system to handle 10,000 image analysis requests per minute?

#### 🛠️ Distributed System Architecture

```
[ 👤 10,000 Users ] ──► [ NGINX Load Balancer ]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [ FastAPI Node 1 ]              [ FastAPI Node 2 ]
    (Stateless API)                 (Stateless API)
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    [ Redis Task Queue ]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     [ GPU Worker Node 1 ]           [ GPU Worker Node 2 ]
     (Triton Server / PyTorch)       (Triton Server / PyTorch)
```

1. **Decouple API Ingestion from AI Inference:**
   - Convert synchronous HTTP processing to an **asynchronous job queue** (Celery + Redis / RabbitMQ or AWS SQS).
   - `POST /api/v1/analyze` accepts the file upload, pushes a job to Redis, and returns `202 Accepted` with a `job_id`.
2. **Dedicated GPU Model Servers (Triton Inference Server):**
   - Offload PyTorch models (YOLOv8 and MiDaS) to dedicated **NVIDIA Triton Inference Servers** with dynamic batching. Dynamic batching combines single incoming requests into GPU batches of 8 or 16 images, increasing throughput by 400%.
3. **Stateless API Scaling:**
   - Run stateless FastAPI nodes behind an NGINX load balancer managed by Kubernetes HPA (Horizontal Pod Autoscaler).

---

### Q22: How do you prevent model drift and maintain accuracy over time in production?

#### 🛠️ MLOps Strategy
1. **Data Flywheel & Human-in-the-Loop (HITL):**
   Allow professional vehicle inspectors using the dashboard to adjust bounding boxes or correct panel labels. Save these user corrections to a secure feedback database.
2. **Dataset Versioning (DVC):**
   Version newly collected edge-case images (e.g., rare car models, unusual lighting) using Data Version Control (DVC).
3. **Shadow Deployment & A/B Testing:**
   Before releasing a new fine-tuned YOLO model to production, deploy it in **Shadow Mode** (runs in parallel on live requests without serving results to users). Evaluate mean Average Precision (mAP) against current models on live data.

---

### Q23: How do you ensure compliance with data privacy regulations (GDPR / CCPA) for vehicle image storage?

#### 🛠️ Privacy Compliance Checklist
1. **Automatic License Plate & Face Anonymization:** Apply an automated OpenCV/YOLO blurring filter over license plates and human faces before saving images to disk or sending them to cloud storage.
2. **Automatic Ephemeral Retention (Auto-Deletion):** Configure cloud storage buckets (AWS S3 / GCP Storage) with **lifecycle policies** that automatically permanently delete raw uploaded images after 30 days.
3. **Zero Location Tracking:** Strip EXIF GPS metadata on ingestion so no geographical tracking data is ever stored in databases.

---

## 💡 Key Technical Cheat-Sheet

| Topic | Technical Implementation |
| :--- | :--- |
| **Backend Framework** | FastAPI + Uvicorn + AnyIO `to_thread.run_sync()` |
| **Detection Model** | Fine-tuned YOLOv8s (`abdullahg7/cardd-yolov8s`) |
| **Classical CV Fallback** | Sobel gradients (dents), Canny edges (scratches), HSV masking (rust) |
| **3D Depth Estimation** | Intel MiDaS inverse depth maps scaled by bounding box dimensions |
| **Scale Calibration** | ISO/IEC 7810 Credit Card reference ($8.56\text{ cm} \times 5.398\text{ cm}$) |
| **API Resilience** | 3-State Circuit Breaker (`CLOSED`, `TRIPPED`, `HALF-OPEN`) |
| **Security Controls** | `secrets.compare_digest` constant-time check, EXIF stripping, 10MB chunking |
| **Frontend Slider** | GPU-accelerated CSS `clip-path` and custom variable `--split-x` |
