# CUDA 12.8.0 (cuDNN runtime variant). Avoid the "cudnn9" suffix – the
# official tag on Docker Hub is just "cudnn-runtime" for cuDNN 9 builds.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# --- Stage 1: Export streaming ONNX (chunked) inside Docker ---
# Use CUDA runtime and pip-install sherpa-onnx (pulls k2 deps) + CPU torch
FROM ${BASE_IMAGE} AS exporter

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    ORT_LOG_SEVERITY_LEVEL=3 \
    ORT_LOG_VERBOSITY_LEVEL=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git git-lfs curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* && \
 git lfs install

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install CPU torch and sherpa-onnx (brings in k2 wheels from their index)
RUN python -m pip install --upgrade pip && \
    pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install 'sherpa-onnx==1.12.14+cuda12.cudnn9' -f https://k2-fsa.github.io/sherpa/onnx/cuda.html && \
    pip install --pre k2 -f https://k2-fsa.github.io/k2/cpu.html && \
    pip install sentencepiece onnx onnxruntime kaldialign lhotse soundfile

WORKDIR /workspace

# Clone exporter and checkpoint, fetch required LFS blobs
RUN git clone --depth=1 https://github.com/k2-fsa/icefall.git icefall-src && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04 ckpt-new && \
    bash -lc "cd ckpt-new/exp && git lfs pull --include 'pretrained.pt' && ln -sf pretrained.pt epoch-99.pt" && \
    bash -lc "cd ckpt-new/data/lang_bpe_500 && git lfs pull --include 'bpe.model'"

# Make icefall importable
RUN pip install -e /workspace/icefall-src
ENV PYTHONPATH=/workspace/icefall-src

# Run export with chunked streaming (chunk=16, left=128)
RUN python3 icefall-src/egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export-onnx.py \
      --exp-dir ./ckpt-new/exp \
      --tokens ./ckpt-new/data/lang_bpe_500/tokens.txt \
      --epoch 99 --avg 1 --use-averaged-model 0 \
      --decode-chunk-len 16 \
      --num-encoder-layers "2,4,3,2,4" \
      --feedforward-dims "1024,1024,2048,2048,1024" \
      --nhead "8,8,8,8,8" \
      --encoder-dims "384,384,384,384,384" \
      --attention-dims "192,192,192,192,192" \
      --encoder-unmasked-dims "256,256,256,256,256" \
      --zipformer-downsampling-factors "1,2,4,8,2" \
      --cnn-module-kernels "31,31,31,31,31" \
      --decoder-dim 512 --joiner-dim 512

# Collect outputs for the runtime image (robust to filename variations)
RUN bash -lc 'set -euo pipefail; \
    out=/out/sherpa-onnx-streaming-zipformer-en-2023-06-21; \
    mkdir -p "$out"; \
    echo "=== Search for exported ONNX files (recursive) ==="; \
    find . -maxdepth 5 -type f -name "*.onnx" -printf "%p\n" || true; \
    get_and_copy() { \
      role=$1; \
      f=$(find . -maxdepth 5 -type f -name "${role}-epoch-99-avg-1-chunk-*.onnx" | head -n1 || true); \
      if [ -z "$f" ]; then \
        f=$(find . -maxdepth 5 -type f -name "${role}-epoch-99-avg-1*.onnx" | head -n1 || true); \
      fi; \
      if [ -z "$f" ]; then \
        echo "FATAL: missing exported ONNX for role=$role"; \
        exit 1; \
      fi; \
      echo "Using $role ONNX: $f"; \
      cp -v "$f" "$out/${role}-epoch-99-avg-1-chunk-16-left-128.onnx"; \
    }; \
    get_and_copy encoder; \
    get_and_copy decoder; \
    get_and_copy joiner; \
    cp -v ckpt-new/data/lang_bpe_500/tokens.txt "$out/"'

# --- Stage 2: Runtime image ---
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

# Models (exported in builder stage)
RUN mkdir -p /models/asr
COPY --from=exporter /out/sherpa-onnx-streaming-zipformer-en-2023-06-21 /models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-21
RUN echo "=== Exported model files ===" && \
    find /models/asr -maxdepth 2 -type f -printf '%P\n' | sort

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
    DECODING_METHOD=greedy_search \
    MAX_ACTIVE_PATHS=8 \
    ENDPOINT_RULE1_MS=800 \
    ENDPOINT_RULE2_MS=400 \
    ENDPOINT_RULE3_MIN_UTT_MS=800 \
    ASR_DIR=/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-21


EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","-s","--"]
# During debugging, keep logs visible if python exits:
# CMD ["bash","-lc","python -O app.py || { echo exit=$?; sleep 3600; }"]
CMD ["python","-O","app.py"]
