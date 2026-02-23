FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgstreamer1.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Download YOLOv8n weights at build time (avoids delay at runtime)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=2", "-Q", "default"]
