# -------------------------------------------------------------
# Dockerfile — Production-grade multi-stage container.
# -------------------------------------------------------------

# Stage 1: Build virtual environment
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final minimal runtime
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Set PyTorch Hub directory inside container to take advantage of Docker volume caching
ENV TORCH_HOME=/app/cache/torch
ENV HF_HOME=/app/cache/huggingface

EXPOSE 8000

# Run uvicorn server in production mode
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
