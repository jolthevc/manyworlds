# Video worker on top of LightX2V's own image (NVFP4 kernels for RTX 5090).
# The base tag comes from the LightWan2.2-A14B model card; confirm it before building.
#   docker build -f docker/worker-video-lightx2v.Dockerfile -t mw-worker-video .
ARG BASE=lightx2v/lightx2v:26052801-cu130-5090
FROM ${BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MW_VIDEO_BACKEND=lightx2v MW_WORKER_KINDS=video.wan_i2v
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir . "huggingface_hub[cli]"
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
