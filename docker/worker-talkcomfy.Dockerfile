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
    MW_COMFY_DIR=/opt/ComfyUI MW_TALK_ENGINE=comfy MW_WORKER_KINDS=talk.infinitetalk,talk.longcat
# git for the clones; ffmpeg muxes and reads video; espeak-ng for misaki/Kokoro fallbacks; libgl/glib for OpenCV;
# gcc because Triton (SageAttention) compiles its GPU driver stub and kernels at run time: without it the first
# sampling step failed with "Failed to find C compiler" (2026-11-22).
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg espeak-ng libgl1 libglib2.0-0 gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
# Keep the base image's torch: a constraint stops ComfyUI's or the wrapper's requirements from replacing it.
# transformers stays on 4.x: the wrapper's wav2vec subclass passes output_hidden_states to the encoder itself,
# which transformers 5 ignores, so MultiTalkWav2VecEmbeds got hidden_states None (2026-11-22, v4 benchmark).
# 4.57.6 is what the research-script image ran 3/3 with.
RUN python -c "import torch; print(f'torch=={torch.__version__.split(\"+\")[0]}')" > /tmp/constraints.txt \
 && echo "transformers==4.57.6" >> /tmp/constraints.txt \
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
# Designed voices (handlers/tts_voxcpm.py, job kinds voice.*): VoxCPM2 (Apache-2.0) and SpeechBrain's ECAPA speaker
# embedding (Apache-2.0). voxcpm goes in without its dependencies: it lists gradio, funasr, modelscope, datasets and
# torchcodec, which only its web demo, training and denoiser use (torchcodec must also match torch exactly); what
# its inference imports is installed here under the same constraints, so torch and transformers 4.57.6 stay put.
ARG VOXCPM=2.0.3
ARG SPEECHBRAIN=1.1.2
RUN pip install --no-cache-dir --no-deps "voxcpm==${VOXCPM}" \
 && pip install --no-cache-dir -c /tmp/constraints.txt einops librosa tqdm pydantic safetensors soundfile \
      "speechbrain==${SPEECHBRAIN}" \
 && python -c "import transformers, torch; assert transformers.__version__ == '4.57.6', transformers.__version__; print(torch.__version__)"
# Original songs (handlers/music_song.py, job kind music.song, family music): ACE-Step 1.5 runs in ComfyUI core (no
# extra package; weights download at the first job); Demucs (MIT code and weights) splits the vocal and instrumental
# stems. Its pins are in docker/constraints-music.txt; torch (and torchaudio, which demucs imports) stay the base
# image's: the image's own torchaudio version joins the constraints when it has one. Not built yet (talkcomfy v15).
COPY docker/constraints-music.txt /tmp/constraints-music.txt
RUN python -c "import torchaudio; print(f'torchaudio=={torchaudio.__version__.split(\"+\")[0]}')" >> /tmp/constraints.txt 2>/dev/null || true \
 && pip install --no-cache-dir -c /tmp/constraints.txt -c /tmp/constraints-music.txt demucs julius dora-search openunmix \
 && python -c "import demucs.apply, demucs.pretrained, julius, pyloudnorm, torch, transformers; assert transformers.__version__ == '4.57.6', transformers.__version__; print(torch.__version__)"
# One continuous talking take (handlers/talk_longcat.py, job kind talk.longcat): LongCat-Video-Avatar 1.6 (MIT) on
# LongCat's own repo at a pinned commit, because ComfyUI-WanVideoWrapper has no 1.6 support yet (WANVIDEO_REF is its
# newest commit, 2026-11-13; 1.6 came out 2026-11-15). Its own environment (/opt/longcat): a venv over the image's
# packages, so it shares torch (and torchaudio) but takes the repo's pins (docker/constraints-longcat.txt:
# transformers 4.41, diffusers 0.35.1, numpy 1.26, audio-separator...) without touching ComfyUI's. The avatar DiT
# needs FlashAttention 2 (no SDPA path): the prebuilt wheel for this image's torch, CUDA 12, C++ ABI and Python.
# Weights are not in the image: ~47 GB (INT8) download at the first job into MW_LONGCAT_DIR (the container disk).
ARG LONGCAT_REF=58eb336edc5c8c63389838727e40b8c18434e54e
ARG FLASH_ATTN=2.8.3
ENV MW_LONGCAT_REPO=/opt/LongCat-Video MW_LONGCAT_PYTHON=/opt/longcat/bin/python MW_LONGCAT_DIR=/models/longcat \
    MW_LONGCAT_COMMIT=${LONGCAT_REF}
COPY docker/constraints-longcat.txt /tmp/constraints-longcat.txt
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /opt/LongCat-Video \
 && git -C /opt/LongCat-Video checkout ${LONGCAT_REF} \
 && rm -rf /opt/LongCat-Video/.git /opt/LongCat-Video/assets \
 && grep -E '^torch(audio)?==' /tmp/constraints.txt > /tmp/constraints-longcat-torch.txt \
 && (python -c "import torchvision; print(f'torchvision=={torchvision.__version__.split(\"+\")[0]}')" >> /tmp/constraints-longcat-torch.txt 2>/dev/null || true) \
 && PY=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")') \
 && TORCH_MM=$(python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))') \
 && ABI=$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")') \
 && python -m venv --system-site-packages /opt/longcat \
 && /opt/longcat/bin/pip install --no-cache-dir -c /tmp/constraints-longcat-torch.txt -c /tmp/constraints-longcat.txt \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN}/flash_attn-${FLASH_ATTN}+cu12torch${TORCH_MM}cxx11abi${ABI}-${PY}-${PY}-linux_x86_64.whl" \
      transformers diffusers numpy loguru einops ftfy psutil av opencv-python imageio imageio-ffmpeg pyarrow \
      scikit-learn scikit-image scipy soundfile soxr librosa "audio-separator[gpu]" pyloudnorm nvidia-ml-py tzdata \
      onnx onnxruntime-gpu cffi chardet \
 && cd /opt/LongCat-Video && /opt/longcat/bin/python -c "import torch, torchvision, flash_attn, transformers, diffusers, audio_separator.separator, longcat_video.pipeline_longcat_video_avatar; assert transformers.__version__ == '4.41.0', transformers.__version__; print('longcat env', torch.__version__)" \
 && python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"
# Lip-sync score (manyworlds/worker/lipsync.py) in its own environment, as in the research-script talk image.
ARG MEDIAPIPE=0.10.21
ARG LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Hand landmarker (Apache-2.0) for the judge's hand close-ups (manyworlds/worker/crops.py), same environment.
ARG HAND_LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
ENV MW_LIPSYNC_PYTHON=/opt/lipsync/bin/python MW_FACE_LANDMARKER=/opt/models/face_landmarker.task \
    MW_HAND_LANDMARKER=/opt/models/hand_landmarker.task
RUN python -m venv /opt/lipsync \
 && /opt/lipsync/bin/pip install --no-cache-dir "mediapipe==${MEDIAPIPE}" "imageio[ffmpeg]" soundfile \
 && mkdir -p /opt/models \
 && python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${LANDMARKER_URL}" /opt/models/face_landmarker.task \
 && test -s /opt/models/face_landmarker.task \
 && python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${HAND_LANDMARKER_URL}" /opt/models/hand_landmarker.task \
 && test -s /opt/models/hand_landmarker.task
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir -c /tmp/constraints.txt .
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
