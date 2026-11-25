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
      transformers diffusers peft numpy loguru einops ftfy psutil av opencv-python imageio imageio-ffmpeg pyarrow \
      scikit-learn scikit-image scipy soundfile soxr librosa "audio-separator[gpu]" pyloudnorm nvidia-ml-py tzdata \
      onnx onnxruntime-gpu cffi chardet \
 && cd /opt/LongCat-Video && /opt/longcat/bin/python -c "import torch, torchvision, flash_attn, transformers, diffusers, audio_separator.separator, longcat_video.pipeline_longcat_video_avatar; assert transformers.__version__ == '4.46.3', transformers.__version__; print('longcat env', torch.__version__)" \
 && python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"
# Singing voice conversion (handlers/music_voice.py, kinds music.convert_voice / music.train_voice, family music; policy
# music.voice, OFF). Each upstream pins torch/transformers below ours, so each gets its own venv over the image's
# packages (--system-site-packages: torch, torchaudio, transformers 4.57.6 come from the image, never reinstalled) and
# only the packages its path imports, filtered from its own pinned requirements at the pinned commit.
#   SoulX-Singer (Apache-2.0 code and weights): preprocess.pipeline (RMVPE F0; vocal_sep and MIDI off) + cli.inference_svc;
#     preprocess.utils hard-imports g2p (ToJyutping, g2pM, g2p_en + nltk). Weights download at the first job.
#   RVC WebUI (MIT code; its assets' licences unverified, policy refuses it): infer/cli.py and train/*.py from the repo
#     root; faiss, parselmouth, ffmpeg-python, av, tensorboard, scikit-learn.
ARG SOULX_REF=81aeb3ae772c70093c3de74dc23c92d983801ae4
# SoulX's llama module calls LlamaAttention the pre-4.48 way (no position_embeddings): its venv takes the repo's own
# transformers pin, shadowing the image's 4.57.6 (first GPU run 2026-11-24: TypeError ... position_embeddings).
ARG SOULX_TRANSFORMERS=4.41.2
ARG RVC_REF=81eed5e8f68b6bed1789f682fe78cdd324495afc
ENV MW_SOULX_DIR=/opt/soulx-singer MW_SOULX_PYTHON=/opt/svc-soulx/bin/python MW_SOULX_TRANSFORMERS=${SOULX_TRANSFORMERS} \
    MW_RVC_DIR=/opt/rvc MW_RVC_PYTHON=/opt/svc-rvc/bin/python
COPY docker/constraints-svc.txt /tmp/constraints-svc.txt
RUN git clone https://github.com/Soul-AILab/SoulX-Singer.git /opt/soulx-singer \
 && git -C /opt/soulx-singer checkout ${SOULX_REF} && rm -rf /opt/soulx-singer/.git \
 && grep -iE '^(accelerate|omegaconf|librosa|scipy|soundfile|tqdm|einops|tojyutping|g2pm|g2p_en|g2p-en|nltk)([=<>~ ]|$)' \
      /opt/soulx-singer/requirements.txt > /tmp/req-soulx.txt \
 && for p in ToJyutping g2pM g2p_en nltk; do grep -qi "^${p}" /tmp/req-soulx.txt || echo "${p}" >> /tmp/req-soulx.txt; done \
 && cat /tmp/req-soulx.txt \
 && python -m venv --system-site-packages /opt/svc-soulx \
 && grep -v '^transformers==' /tmp/constraints.txt > /tmp/constraints-soulx-base.txt \
 && /opt/svc-soulx/bin/pip install --no-cache-dir -c /tmp/constraints-soulx-base.txt -c /tmp/constraints-svc.txt -r /tmp/req-soulx.txt \
      "transformers==${SOULX_TRANSFORMERS}" \
 && cd /opt/soulx-singer && /opt/svc-soulx/bin/python -c "import preprocess.pipeline, soulxsinger.models.soulxsinger_svc, transformers, torch; assert transformers.__version__ == '${SOULX_TRANSFORMERS}', transformers.__version__; print('soulx ok', torch.__version__)" \
 && python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /opt/rvc \
 && git -C /opt/rvc checkout ${RVC_REF} && rm -rf /opt/rvc/.git \
 && grep -iE '^(faiss-cpu|librosa|praat-parselmouth|scipy|soundfile|ffmpeg-python|av|tensorboard|scikit-learn|tqdm)([=<>~ ]|$)' \
      /opt/rvc/requirments_cu128_py312.txt > /tmp/req-rvc.txt \
 && for p in faiss-cpu praat-parselmouth ffmpeg-python av tensorboard scikit-learn; do grep -qi "^${p}" /tmp/req-rvc.txt || echo "${p}" >> /tmp/req-rvc.txt; done \
 && cat /tmp/req-rvc.txt \
 && python -m venv --system-site-packages /opt/svc-rvc \
 && /opt/svc-rvc/bin/pip install --no-cache-dir -c /tmp/constraints.txt -c /tmp/constraints-svc.txt -r /tmp/req-rvc.txt \
 && mkdir -p /opt/rvc/assets/weights /opt/rvc/assets/indices \
 && cd /opt/rvc && /opt/svc-rvc/bin/python -c "import faiss, parselmouth, av, ffmpeg, sklearn, torch.utils.tensorboard, transformers; print('rvc ok')"
# Named graphs (job kind comfy.graph, worker/handlers/comfy_graph.py, policy comfy_graphs). LTX-2.5 needs no pack:
# its nodes are in ComfyUI core (templates want 0.32.0+; COMFYUI_REF is v0.45.0) and its int8 ConvRot weights run on
# comfy-kitchen, which ComfyUI's requirements pin (0.2.41). EchoMimicV3-Flash runs in smthemex/ComfyUI_EchoMimic
# (Apache-2.0) at a pinned commit, with only the packages its V3_flash path imports (docker/constraints-comfygraphs.txt);
# its weights download at the first job into the weights cache, reached through ComfyUI/models/echo_mimic (a link).
# Not built yet: the smoke check (check_talkcomfy) validates every named graph against ComfyUI's node catalogue.
ARG ECHOMIMIC_REF=9f430d9219c378505dedec6ed1749d5088745b3c
COPY docker/constraints-comfygraphs.txt /tmp/constraints-comfygraphs.txt
RUN git clone https://github.com/smthemex/ComfyUI_EchoMimic.git /opt/ComfyUI/custom_nodes/ComfyUI_EchoMimic \
 && git -C /opt/ComfyUI/custom_nodes/ComfyUI_EchoMimic checkout ${ECHOMIMIC_REF} \
 && rm -rf /opt/ComfyUI/custom_nodes/ComfyUI_EchoMimic/.git \
 && pip install --no-cache-dir -c /tmp/constraints.txt -c /tmp/constraints-comfygraphs.txt \
      moviepy omegaconf decord matplotlib loguru ffmpeg-python \
 && python -c "import moviepy, omegaconf, decord, matplotlib, loguru, ffmpeg, diffusers, accelerate, cv2, librosa, pyloudnorm, transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"
# Expressive voices and automatic casting (handlers/tts_engines.py, job kinds voice.speak.indextts, voice.speak.qwen3,
# voice.cast; worker family tts, talkcomfy v21). Each engine pins transformers apart from the image's 4.57.6, so each
# gets its own venv over the image's packages (--system-site-packages: torch and torchaudio come from the image and a
# constraint keeps them) and runs as a small server (manyworlds/worker/tts_server.py):
#   IndexTTS-2.5 (bilibili Model Use License: commercial use below 100M MAU / RMB 1bn revenue; its outputs must never
#     train another model) at a pinned commit, with the repo's own pins (docker/constraints-indextts.txt). Weights
#     (~8 GB) download at the first job into MW_INDEXTTS_DIR on the container disk; tts_server.py fetches w2v-BERT 2.0,
#     CAMPPlus and BigVGAN itself so the repo's downloader never fetches MaskGCT's CC-BY-NC semantic codec.
#   Qwen3-TTS 1.7B Base and VoiceDesign (Apache-2.0), qwen-tts 0.1.1 without its gradio demo
#     (docker/constraints-qwentts.txt); weights at the first job into HF_HOME.
# The judges run in the image's own environment: UTMOS22 strong (MIT; tarepan/SpeechMOS v1.2.0 and its checkpoint
# fetched here into MW_TORCH_HUB, so no machine downloads code at run time), and two wav2vec2 classifiers through
# transformers (gender, Apache-2.0; age, Apache-2.0) downloaded at the first job.
ARG INDEXTTS_REF=d9e41aac89fd00b3d71497fddb287b7f24613712
ARG SPEECHMOS_REF=v1.2.0
ENV MW_INDEXTTS_REPO=/opt/index-tts MW_INDEXTTS_PYTHON=/opt/indextts/bin/python MW_INDEXTTS_DIR=/root/indextts-2.5 \
    MW_QWENTTS_PYTHON=/opt/qwentts/bin/python MW_TORCH_HUB=/opt/torch-hub MW_INDEXTTS_COMMIT=${INDEXTTS_REF}
COPY docker/constraints-indextts.txt /tmp/constraints-indextts.txt
COPY docker/constraints-qwentts.txt /tmp/constraints-qwentts.txt
RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-fmt-all && rm -rf /var/lib/apt/lists/* \
 && grep -E '^torch(audio)?==' /tmp/constraints.txt > /tmp/constraints-engines-torch.txt \
 && git clone https://github.com/index-tts/index-tts.git /opt/index-tts \
 && git -C /opt/index-tts checkout ${INDEXTTS_REF} \
 && rm -rf /opt/index-tts/.git \
 && python -m venv --system-site-packages /opt/indextts \
 && /opt/indextts/bin/pip install --no-cache-dir -c /tmp/constraints-engines-torch.txt -c /tmp/constraints-indextts.txt \
      /opt/index-tts "huggingface_hub[cli]" \
 && /opt/indextts/bin/python -c "import inspect, torch, transformers; from indextts.infer_v2_5 import IndexTTS2; assert transformers.__version__ == '4.52.1', transformers.__version__; assert 'emo_vector' in inspect.signature(IndexTTS2.infer).parameters; print('indextts env', torch.__version__)" \
 && python -m venv --system-site-packages /opt/qwentts \
 && /opt/qwentts/bin/pip install --no-cache-dir -c /tmp/constraints-engines-torch.txt -c /tmp/constraints-qwentts.txt \
      --no-deps qwen-tts \
 && /opt/qwentts/bin/pip install --no-cache-dir -c /tmp/constraints-engines-torch.txt -c /tmp/constraints-qwentts.txt \
      transformers accelerate librosa soundfile sox onnxruntime einops \
 && /opt/qwentts/bin/python -c "import torch, transformers; from qwen_tts import Qwen3TTSModel; assert transformers.__version__ == '4.57.3', transformers.__version__; print('qwentts env', torch.__version__)" \
 && mkdir -p /opt/torch-hub \
 && python -c "import sys, torch; torch.hub.set_dir(sys.argv[1]); torch.hub.load(sys.argv[2], 'utmos22_strong', trust_repo=True); print('utmos22 ok')" \
      /opt/torch-hub "tarepan/SpeechMOS:${SPEECHMOS_REF}" \
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
