# Yap Sherpa-ONNX GPU Streaming ASR Server

GPU streaming ASR based on Sherpa-ONNX Zipformer RNNT (English) with batched GPU decoding. A single WebSocket endpoint accepts `s16le` 16 kHz mono frames and returns JSON partial/final messages.

## What’s inside
- Streaming English Zipformer RNNT: `sherpa-onnx-streaming-zipformer-en-2023-06-26`
- CUDA 12.x runtime + cuDNN 9 wheel (installed via pip); GPU-enabled `sherpa-onnx`
- Batched decode across ready streams for high throughput (L40S-ready)

## Quick start (Docker, GPU)

```bash
# Build
docker build -t sherpa-asr:gpu -f docker/Dockerfile .

# Run (all GPUs). Pin with CUDA_VISIBLE_DEVICES if needed.
docker run --rm --gpus all -p 8000:8000 \
  -e PROVIDER=cuda \
  -e MAX_BATCH=64 \
  -e PARTIAL_HZ=20 \
  --name sherpa-asr sherpa-asr:gpu
```

Server starts at `ws://0.0.0.0:8000/ws`.

## WebSocket protocol
- Send binary frames: PCM16LE mono at 16 kHz (any chunking; 20 ms recommended)
- Terminate stream with control frame: `b"__CTRL__:EOS"`
- Receive JSON text frames:
  - Partial: `{ "type": "partial", "text": "..." }` (no punctuation)
  - Final: `{ "type": "final", "text": "..." }` (stable segment text)

## Config (env vars)
- `WS_HOST` (default `0.0.0.0`)
- `WS_PORT` (default `8000`)
- `SAMPLE_RATE` (default `16000`)
- `PROVIDER` (`cuda` or `cpu`, default `cuda`)
- `MAX_BATCH` (default `64`)
- `MAX_CONNECTIONS` (default `2048`)
- `PARTIAL_HZ` (partials per second per client; default `20`)
- `ENDPOINT_RULE1_MS`/`RULE2_MS`/`RULE3_MIN_UTT_MS` (default `800/400/800`)
- `ASR_DIR` (default `/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26`)

## Included models
The container downloads and unpacks the ASR model at build time under `/models`.

## Smoke tests (included clients)

```bash
# Warm (sends fast to warm kernels)
WS=ws://127.0.0.1:8000/ws python tests/warmup.py --file samples/mid.wav --rtf 10 --print-partials

# Single-file smoke
WS=ws://127.0.0.1:8000/ws python tests/client.py --file samples/mid.wav --print-partials --full-text

# Concurrency bench (synthetic tone)
WS=ws://127.0.0.1:8000/ws python tests/bench.py --streams 64 --duration 30 --frame-ms 20 --rtf 1.0
```

The `samples/` directory is copied into the image and available at `/app/samples`.

## Why this stack
- Sherpa-ONNX Zipformer RNNT: optimized for streaming; exposes `decode_streams()` for batch GPU decode
- CUDA 12.x + cuDNN 9 wheels: straightforward L40S deployment

## Notes & tuning
- Lower latency: increase `PARTIAL_HZ` (e.g., 30–40) and/or tighten endpointing
- Higher throughput: raise `MAX_BATCH` if GPU has headroom
- CPU fallback: set `PROVIDER=cpu` (slower)

## Build & push to Docker Hub

```bash
# Login once
docker login

# Choose your repo and tag
export DOCKER_IMAGE=<your-dockerhub-user>/sherpa-asr
export DOCKER_TAG=latest

# Build and push with buildx (amd64 by default)
bash docker/publish.sh

# Verify
docker buildx imagetools inspect ${DOCKER_IMAGE}:${DOCKER_TAG}
```

If the default CUDA base image tag is unavailable in your registry mirror, override it:

```bash
export DOCKER_IMAGE=<your-dockerhub-user>/sherpa-asr
export DOCKER_TAG=latest
# Example: override base image if your mirror lacks defaults
export DOCKER_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
bash docker/publish.sh
```

## Sources
- Sherpa-ONNX GPU install (CUDA 12.8 + cuDNN 9): `https://k2-fsa.github.io/sherpa/onnx/python/install.html`
- Streaming Zipformer (EN): `https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html#streaming-zipformer-en-2023-06-26`
