# GPU worker image, one per job family:
#   docker build -f docker/worker.Dockerfile --build-arg EXTRA=image -t mw-worker-image .
#   docker build -f docker/worker.Dockerfile --build-arg EXTRA=tts   -t mw-worker-tts .
# CUDA 12.8+ is required for RTX 5090 (Blackwell); it also runs on 4090s.
# Bake weights in with PREFETCH so interruptible nodes don't re-download ~10-60 GB on
# every cold start (SaladCloud nodes start from the image).
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}
ARG EXTRA=image
ARG PREFETCH=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir ".[${EXTRA}]" "huggingface_hub[cli]"
COPY policy ./policy
RUN if [ -n "$PREFETCH" ]; then for m in $PREFETCH; do huggingface-cli download "$m"; done; fi
ENTRYPOINT ["mw"]
CMD ["worker"]
