# LoRA training worker (ostris/ai-toolkit). Needs a 24 GB card (RTX 3090/4090/A5000).
#   docker build -f docker/worker-train.Dockerfile -t mw-worker-train .
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}
ARG AI_TOOLKIT_REF=main
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf \
    MW_AI_TOOLKIT=/ai-toolkit MW_WORKER_KINDS=train.lora
# libgl1/libglib2.0-0: OpenCV (imported by ai-toolkit via diffusers' watermarker) needs them.
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${AI_TOOLKIT_REF} https://github.com/ostris/ai-toolkit.git /ai-toolkit \
 && pip install --no-cache-dir -r /ai-toolkit/requirements.txt
WORKDIR /app
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir .
COPY policy ./policy
ENTRYPOINT ["mw"]
CMD ["worker"]
