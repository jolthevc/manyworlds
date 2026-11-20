# Lip-synced talking video worker (MeiGen-AI/InfiniteTalk, Apache 2.0).
#   scripts/build-worker.sh talk v1
# CUDA 12.8 base so RTX 5090s work. InfiniteTalk's README pins torch 2.4.1 / cu121; we keep the base's
# torch 2.8 (a constraint stops its requirements from replacing it) with the matching xformers and a
# prebuilt flash-attn wheel (2.8.3: the only torch-2.8 wheel; xformers accepts up to 2.8.2 by number only,
# hence the version-check override). The smoke test runs its script's --help, which imports all of it.
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}
ARG INFINITETALK_REF=50aa0a94184315407a991ae804d9b58d6d311ba8
ARG FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf \
    MW_INFINITETALK_REPO=/opt/InfiniteTalk MW_WORKER_KINDS=talk.infinitetalk \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
# libgl1/libglib2.0-0 for OpenCV; espeak-ng for misaki/Kokoro fallbacks; ffmpeg muxes the audio.
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg espeak-ng libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /opt/InfiniteTalk \
 && git -C /opt/InfiniteTalk checkout ${INFINITETALK_REF} && rm -rf /opt/InfiniteTalk/.git \
 && sed -i '/^from inspect import ArgSpec$/d' /opt/InfiniteTalk/wan/multitalk.py \
 && sed -i 's|Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True)|Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager")|' /opt/InfiniteTalk/generate_infinitetalk.py \
 && grep -q 'attn_implementation="eager"' /opt/InfiniteTalk/generate_infinitetalk.py
# Written for Python 3.10 and transformers 4.49+: the ArgSpec import is unused (3.11 removed it), and its wav2vec
# wrapper sets config.output_attentions on every forward pass, which newer transformers refuses under SDPA
# attention (2026-11-20, first benchmark): load it with eager attention, and keep transformers on 4.x.
# gradio and dashscope serve its demo app only.
RUN python -c "import torch; print(f'torch=={torch.__version__.split(\"+\")[0]}')" > /tmp/constraints.txt \
 && echo "transformers==4.57.6" >> /tmp/constraints.txt \
 && grep -vE '^(gradio|dashscope)' /opt/InfiniteTalk/requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt xformers==0.0.32.post2 "${FLASH_ATTN_WHL}" \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/req.txt \
      "misaki[en]" librosa soundfile "kokoro>=0.9" pyloudnorm "brotli>=1.2" "huggingface_hub[cli]"
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir -c /tmp/constraints.txt .
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
