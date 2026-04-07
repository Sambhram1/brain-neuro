# ── Stage 1: Build frontend ──
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + model ──
FROM python:3.11-slim

# System dependencies (ffmpeg for whisperx, libsndfile for audio, git for pip installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn fastapi python-multipart yt-dlp \
    && pip install --no-cache-dir "whisperx @ git+https://github.com/m-bain/whisperX.git" faster-whisper

# Copy backend code and model config
COPY backend.py .
COPY model_weights/ ./model_weights/

# Copy pre-built frontend into static/
COPY --from=frontend-build /app/frontend/dist/ ./static/

# Environment
ENV HF_HOME=/app/cache
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
