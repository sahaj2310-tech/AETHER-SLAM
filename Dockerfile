# ==========================================
# Stage 1: Build React Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: CUDA Runtime Backend + Unified Static Serve
# ==========================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-opencv \
        libgl1-mesa-glx \
        libglib2.0-0 \
        portaudio19-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first; requirements.txt no longer lists torch
# so there's no duplicate install/version war.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend
COPY tip/ ./tip
COPY yolov8n-seg.pt ./yolov8n-seg.pt

# Built frontend assets
COPY --from=frontend-builder /app/frontend/dist ./backend/static

EXPOSE 8000
# Replay a recorded sequence instead of the webcam:  -e AETHER_SOURCE=/data/rgbd_dataset_freiburg1_desk

CMD ["python", "-m", "backend.main"]
