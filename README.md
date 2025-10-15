# Yap Vosk GPU Streaming ASR Server

GPU-accelerated speech-to-text server built around Vosk (Kaldi + CUDA bindings). Audio arrives over a WebSocket connection, partial hypotheses stream in real time, and the final transcript flushes as soon as the client emits an `__CTRL__:EOS` control frame.

## Highlights
- Vosk CUDA decoder with the `vosk-model-en-us-0.22` English pack baked into the container
- Binary WebSocket protocol (`s16le` mono 16 kHz) with partial + final JSON messages
- Stateless asyncio server that supports dozens of concurrent sessions (`CONCURRENCY` env)
- Minimal runtime footprint: Python 3.10, websockets, numpy, soundfile, uvloop
- Ready-to-run Docker image (`docker/Dockerfile`) for GPU deployment, plus lightweight CLI clients for smoke tests and benchmarks

## Quick Start (Docker)

```bash
# Build the GPU-enabled image (override MODEL_URL/MODEL_NAME to swap models)
docker build \
  -t vosk-gpu-ws:latest \
  -f docker/Dockerfile .

# If you are on Apple Silicon or need an x86_64 image locally, use buildx
# and pin the platform (the --load flag imports the image into your local daemon):
# docker buildx build --platform=linux/amd64 -t vosk-gpu-ws:latest -f docker/Dockerfile . --load

# Launch on a CUDA host (NVIDIA Container Toolkit required)
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e CONCURRENCY=64 \
  vosk-gpu-ws:latest
```

### Notes on GPU base image

- Base image: `alphacep/kaldi-vosk-server-gpu:latest` (bundles CUDA-enabled Kaldi/Vosk, including `libvosk` with GPU support). We intentionally do not `pip install vosk`; doing so would replace the GPU build with the CPU wheel.
- To change the port inside the container, set `PORT` (our server) or if you use the Alphacep WS server directly, `VOSK_SERVER_PORT` (default 2700).
- Verify GPU usage during decoding with: `nvidia-smi dmon -s u`.

### Build & push to your Docker Hub registry

```bash
export DOCKER_IMAGE=my-dockerhub-user/vosk-gpu-ws
export DOCKER_TAG=prod        # optional, defaults to "latest"
# Optional: override target platform (defaults to linux/amd64)
# export DOCKER_PLATFORM=linux/arm64
bash docker/publish.sh
```

The helper script uses `docker buildx build --platform=${DOCKER_PLATFORM:-linux/amd64}` and pushes the image, so it automatically targets Linux/x86_64 by default. Override `MODEL_URL` / `MODEL_NAME` in the environment before running if you want to embed a different acoustic model. To publish for a different architecture, set `DOCKER_PLATFORM` (e.g. `linux/arm64`).

Verify what was built:

```bash
docker image inspect ${DOCKER_IMAGE}:${DOCKER_TAG:-latest} --format '{{.Architecture}}/{{.Os}}'
```

The server starts immediately after loading the Vosk model and listens on `ws://0.0.0.0:8000`.

### Deploying a different model
- At build time: `docker build --build-arg MODEL_URL=<zip> --build-arg MODEL_NAME=<top-level-folder> ...`
- At runtime: mount a directory into `/models/en` and export `MODEL_DIR` to point at the unpacked model root

```bash
# Example: mount a custom model from the host
 docker run --rm -it --gpus all \
   -p 8000:8000 \
   -v /path/to/model:/custom-model \
   -e MODEL_DIR=/custom-model \
   vosk-gpu-ws:latest
```

## Configuration
All knobs are exposed via environment variables (same in Docker and bare-metal):
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)
- `MODEL_DIR` (default `/models/en`)
- `SAMPLE_RATE` (default `16000` Hz)
- `CONCURRENCY` (max simultaneous sessions, default `64`)
- `MAX_WS_MESSAGE_SIZE` (bytes; default `8388608`)
- `ENABLE_WORD_TIMES` (`1`/`0`; default `1`)
- `LOG_LEVEL` (default `INFO`)
- `VOSK_LOG_LEVEL` (default `-1` → suppress Vosk debug output)

The server initializes the GPU path at startup via `GpuInit()` before loading the model. If initialization fails, it logs a warning and continues (CPU path), but on the GPU base this should succeed when a CUDA device is present.

## WebSocket Protocol
- Endpoint: `ws://<host>:<port>`
- Send raw PCM16 mono frames at 16 kHz; any chunk size works (20 ms suggested)
- After the final frame, transmit the control bytes `b"__CTRL__:EOS"`
- Optional reset without reconnecting: `b"__CTRL__:RESET"`
- JSON text frames streamed back:
  - Partial hypothesis: `{ "type": "partial", "text": "..." }`
  - Final hypothesis: `{ "type": "final", "text": "..." }`

Connections stay open until the client closes them or the server receives `EOS`.

## Client Utilities
Install the Python requirements locally (or reuse the Docker container):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then use the helper scripts under `tests/`:
- `tests/client.py` – smoke test against a live server
- `tests/warmup.py` – measures latency, time-to-first-word, and throughput
- `tests/bench.py` – synthetic concurrency benchmark (sine wave generator)

Each tool honours the `--url` flag (default `ws://127.0.0.1:8000`). Audio samples live under `samples/`.

## Development Notes
- The runtime automatically installs `uvloop` when available; fallbacks to asyncio otherwise
- Logging uses a simple `%(asctime)s %(levelname)s %(name)s: %(message)s` format for compatibility with structured collectors
- The server is stateless; horizontal scale-out is achieved by running multiple containers behind your load balancer

## Troubleshooting
- **`vosk.Model` fails to load** → ensure `MODEL_DIR` points at the unpacked model folder (the one containing `am`, `conf`, etc.) and that the Docker volume mount includes read permissions
- **No partials come back** → confirm you are streaming `int16` little-endian PCM and not base64/text frames
- **High latency** → lower the beam size of your custom model, decrease chunk size, or raise `CONCURRENCY` to window more simultaneous sessions

Feel free to tailor the Docker CMD or wrap `server.ws_server:run` inside your preferred process supervisor for production deployments.
