FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    curl ca-certificates ffmpeg tini && \
    rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# GPU-enabled sherpa-onnx (CUDA 12.x + cuDNN9), plus runtime libs
RUN python -m pip install --upgrade pip && \
    pip install 'sherpa-onnx==1.12.13+cuda12.cudnn9' \
      -f https://k2-fsa.github.io/sherpa/onnx/cuda.html && \
    pip install websockets uvloop numpy

# Models
RUN mkdir -p /models/asr /models/punct
# ASR: Streaming Zipformer (English)
RUN curl -L -o /models/asr/model.tar.bz2 \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2 \
 && tar -C /models/asr -xf /models/asr/model.tar.bz2 \
 && rm /models/asr/model.tar.bz2
# Punctuation + casing (English)
RUN curl -L -o /models/punct/punct.tar.bz2 \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2 \
 && tar -C /models/punct -xf /models/punct/punct.tar.bz2 \
 && rm /models/punct/punct.tar.bz2

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
    ASR_DIR=/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
    PUNCT_DIR=/models/punct/sherpa-onnx-online-punct-en-2024-08-06

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python","-O","app.py"]
