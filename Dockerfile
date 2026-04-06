FROM python:3.10-slim

# System deps: ffmpeg, git, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install CPU-only PyTorch first (saves ~2GB vs default CUDA build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install tribev2 and remaining deps in separate layers for better caching
RUN pip install --no-cache-dir \
    "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    yt-dlp \
    pydantic \
    numpy \
    huggingface_hub \
    uv

# Copy app code
COPY --chown=user:user backend.py .
COPY --chown=user:user model_weights/ ./model_weights/
COPY --chown=user:user static/ ./static/

# Switch to non-root user
USER user

# HF Spaces exposes port 7860
ENV PORT=7860
EXPOSE 7860

# HF_TOKEN is set via Space secrets — tribev2 needs it for LLaMA access
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
ENV HF_HOME=/home/user/.cache/huggingface

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "600"]
