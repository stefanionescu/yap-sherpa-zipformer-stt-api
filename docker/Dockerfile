# CUDA 12.8.0 (cuDNN runtime variant). Avoid the "cudnn9" suffix – the
# official tag on Docker Hub is just "cudnn-runtime" for cuDNN 9 builds.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    SHERPA_ONNX_LOG=1 \
    ORT_LOG_SEVERITY_LEVEL=1 \
    ORT_LOG_VERBOSITY_LEVEL=1

# System deps (+ bzip2/lbzip2: needed for .tar.bz2 archives)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    curl ca-certificates ffmpeg tini \
    bzip2 lbzip2 xz-utils \
 && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# sherpa-onnx with CUDA (includes a matching ORT inside their wheel)
RUN python -m pip install --upgrade pip && \
    pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    pip install 'sherpa-onnx==1.12.14+cuda12.cudnn9' \
      -f https://k2-fsa.github.io/sherpa/onnx/cuda.html && \
    pip install websockets uvloop numpy soundfile

# Models
RUN mkdir -p /models/asr
# ASR: Streaming Zipformer (English)
RUN curl -L -o /models/asr/model.tar.bz2 \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2 \
 && tar -C /models/asr -xjf /models/asr/model.tar.bz2 \
 && rm /models/asr/model.tar.bz2 \
 && echo "=== Model files extracted ===" \
 && find /models/asr -maxdepth 2 -type f -printf '%P\n' | sort

# App
WORKDIR /app
COPY server/ /app/
COPY tests /app/tests
COPY samples /app/samples

# Defaults; tune at run time via -e
ENV WS_HOST=0.0.0.0 \
    WS_PORT=8000 \
    SAMPLE_RATE=16000 \
    PROVIDER=cuda \
    MAX_BATCH=64 \
    MAX_CONNECTIONS=2048 \
    PARTIAL_HZ=20 \
    ENDPOINT_RULE1_MS=800 \
    ENDPOINT_RULE2_MS=400 \
    ENDPOINT_RULE3_MIN_UTT_MS=800 \
    ASR_DIR=/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26


EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","-s","--"]
# During debugging, keep logs visible if python exits:
# CMD ["bash","-lc","python -O app.py || { echo exit=$?; sleep 3600; }"]
CMD ["python","-O","app.py"]
