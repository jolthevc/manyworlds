# Lip-synced talking video worker (MeiGen-AI/InfiniteTalk, Apache 2.0).
#   scripts/build-worker.sh talk v1
# CUDA 12.8 base so RTX 5090s work. InfiniteTalk's README pins torch 2.4.1 / cu121; we keep the base's
# torch 2.8 (a constraint stops its requirements from replacing it) with the matching xformers and a
# prebuilt flash-attn wheel (2.8.3: the only torch-2.8 wheel; xformers accepts up to 2.8.2 by number only,
# hence the version-check override). The smoke test runs its script's --help, which imports all of it.
# devel, not runtime: InfiniteTalk's fp8 build runs on optimum-quanto's Marlin kernel, which is compiled
# below at build time with MAX_JOBS=2 (full parallelism ran the laptop's Docker out of memory) (a runtime image has no nvcc: 'CUDA_HOME is not set' on the first A40, 2026-11-20).
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE}
ARG INFINITETALK_REF=50aa0a94184315407a991ae804d9b58d6d311ba8
ARG FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf \
    MW_INFINITETALK_REPO=/opt/InfiniteTalk MW_WORKER_KINDS=talk.infinitetalk \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0" QUANTO_CUDA_ARCH=800
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
# Compile quanto's CUDA kernels now, for the GPUs we rent (A40/3090: 8.6, 4090: 8.9, 5090: 12.0), so no machine compiles
# at run time. Quanto derives its arch flag from the GPUs it sees (none here); pinning it (QUANTO_CUDA_ARCH, also
# set at run time) keeps the build flags identical, so the cached build is reused instead of rebuilt.
RUN Q=$(python -c "import optimum.quanto, os; print(os.path.dirname(optimum.quanto.__file__))")/library/extensions/cuda/__init__.py \
 && sed -i 's|^quanto_cuda_arch = get_max_cuda_arch()$|quanto_cuda_arch = os.environ.get("QUANTO_CUDA_ARCH") or get_max_cuda_arch()|' $Q \
 && grep -q 'os.environ.get("QUANTO_CUDA_ARCH")' $Q \
 && MAX_JOBS=2 python -c "from optimum.quanto.library.extensions.cuda import ext; ext.lib" \
 && ls $(dirname $Q)/build/*.so
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir -c /tmp/constraints.txt .
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
