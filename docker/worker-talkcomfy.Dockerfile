# Lip-synced talking video on ComfyUI + Kijai's WanVideoWrapper (manyworlds/worker/comfy.py, handlers/talk_comfy.py).
#   git tag worker-talkcomfy-v1 && git push origin worker-talkcomfy-v1      (built and smoke-tested on Depot)
# The community's stack for InfiniteTalk: fp8-scaled model + adapter, the 4-step lightx2v LoRA and SageAttention, on
# 24-48 GB cards. InfiniteTalk's own research script (docker/worker-talk.Dockerfile) was slow or out of memory on one
# card (2026-11-22). ComfyUI (GPL-3.0) runs as a local server inside this worker; WanVideoWrapper is Apache-2.0.
# Weights are not in the image: they download once to MW_MODEL_DIR/comfy (the pool's network volume if it has one).
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}
ARG COMFYUI_REF=v0.45.0
ARG WANVIDEO_REF=a7439db30bed0c5b5fa437d77ae8a7bb220dfd00
ARG SAGEATTENTION=1.0.6
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf \
    MW_COMFY_DIR=/opt/ComfyUI MW_TALK_ENGINE=comfy MW_WORKER_KINDS=talk.infinitetalk
# git for the clones; ffmpeg muxes and reads video; espeak-ng for misaki/Kokoro fallbacks; libgl/glib for OpenCV;
# gcc because Triton (SageAttention) compiles its GPU driver stub and kernels at run time: without it the first
# sampling step failed with "Failed to find C compiler" (2026-11-22).
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg espeak-ng libgl1 libglib2.0-0 gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
# Keep the base image's torch: a constraint stops ComfyUI's or the wrapper's requirements from replacing it.
RUN python -c "import torch; print(f'torch=={torch.__version__.split(\"+\")[0]}')" > /tmp/constraints.txt \
 && git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout ${COMFYUI_REF} \
 && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
 && git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper checkout ${WANVIDEO_REF} \
 && rm -rf /opt/ComfyUI/.git /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/.git \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /opt/ComfyUI/requirements.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt "sageattention==${SAGEATTENTION}" \
      "misaki[en]" soundfile "kokoro>=0.9" pyloudnorm "brotli>=1.2" "huggingface_hub[cli]" imageio imageio-ffmpeg \
      "runpod>=1.7,<2"
# Lip-sync score (manyworlds/worker/lipsync.py) in its own environment, as in the research-script talk image.
ARG MEDIAPIPE=0.10.21
ARG LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
ENV MW_LIPSYNC_PYTHON=/opt/lipsync/bin/python MW_FACE_LANDMARKER=/opt/models/face_landmarker.task
RUN python -m venv /opt/lipsync \
 && /opt/lipsync/bin/pip install --no-cache-dir "mediapipe==${MEDIAPIPE}" "imageio[ffmpeg]" soundfile \
 && mkdir -p /opt/models \
 && python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${LANDMARKER_URL}" /opt/models/face_landmarker.task \
 && test -s /opt/models/face_landmarker.task
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir -c /tmp/constraints.txt .
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
