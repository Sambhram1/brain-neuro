FROM python:3.10

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg git curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (much smaller than CUDA default)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# tribev2 from source
RUN pip install --no-cache-dir \
    "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"

# App dependencies
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" yt-dlp pydantic numpy huggingface_hub uv

# Copy app code
COPY --chown=user:user backend.py .
COPY --chown=user:user model_weights/ ./model_weights/
COPY --chown=user:user static/ ./static/

USER user

ENV PORT=7860
EXPOSE 7860
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
ENV HF_HOME=/home/user/.cache/huggingface

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "600"]
