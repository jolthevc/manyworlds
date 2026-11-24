# Control plane and CPU-only fake worker.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The finished file's lip-sync score (manyworlds/review/sync.py, extra `sync`): MediaPipe (Apache-2.0 code and model)
# on the CPU. Its OpenCV needs libGL and glib at import. The face landmarker model (a few MB) is pinned by its
# versioned path, as in the talk worker images.
ARG LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
ENV MW_FACE_LANDMARKER=/opt/models/face_landmarker.task
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/models \
 && python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${LANDMARKER_URL}" /opt/models/face_landmarker.task \
 && test -s /opt/models/face_landmarker.task
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir ".[test,sync]"
COPY policy ./policy
COPY tests ./tests
ENTRYPOINT ["mw"]
CMD ["status"]
