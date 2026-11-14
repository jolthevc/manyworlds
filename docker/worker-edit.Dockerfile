# CPU worker for edits (narrated videos): Kokoro TTS on CPU + ffmpeg. No GPU needed.
#   docker build -f docker/worker-edit.Dockerfile -t mw-worker-edit .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir ".[edit]"
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
