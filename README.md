# Yap Sherpa-ONNX GPU Streaming ASR Server

GPU streaming ASR based on Sherpa-ONNX Zipformer RNNT (English) with batched GPU decoding. A single WebSocket endpoint accepts `s16le` 16 kHz mono frames and returns JSON partial/final messages.

## What's inside
- Streaming English Zipformer RNNT: exported at build time (chunked, `chunk=16`)
- Output directory (inside image): `/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-21`
- **sherpa-onnx 1.12.14+cuda12.cudnn9** (CUDA-enabled wheel with bundled ORT) for GPU support
- Uses **factory method API** (`OnlineRecognizer.from_transducer()`), the stable public interface
- Batched decode across ready streams for high throughput (L40S-ready)
- **Enforces chunked streaming exports** for optimal real-time performance

## Model Quality vs Performance Trade-off

**Current default (2023-06-21):**
- **Pros:** Higher transcription quality, better handling of natural speech
- **Cons:** No chunked streaming exports, poor real-time performance (RTF > 1.0)

**Alternative (2023-06-26):**
- **Pros:** Has chunked streaming exports, achieves real-time performance (RTF < 0.5)
- **Cons:** Lower transcription quality, especially with filler words and disfluencies

The server will **fail fast** if you try to use a model pack without chunked streaming exports. This repo now performs the export **inside Docker during build** from the HF checkpoint `marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04` using Icefall's exporter (Zipformer S7, multi). The generated files include `encoder-epoch-99-avg-1-chunk-16-left-128.onnx`, `decoder-...`, `joiner-...`, and `tokens.txt`.

## Breaking Changes in sherpa-onnx 1.12.13+

**If you're upgrading from older versions or encountering API errors:**

The sherpa-onnx packaging changed significantly around 1.12.13. Some wheel builds (especially custom CUDA ones like `1.12.13+cuda12.cudnn9`) **don't re-export** config classes at the top level, causing errors like:
- `OnlineRecognizer() takes no arguments`
- `module 'sherpa_onnx' has no attribute 'OnlineRecognizerConfig'`

**Solution:** This project now uses the **stable public API**:
```python
# ✅ WORKS: Factory method (stable public API)
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="tokens.txt", encoder="encoder.onnx", decoder="decoder.onnx", joiner="joiner.onnx", 
    provider="cuda", sample_rate=16000, feature_dim=80, ...
)

# ❌ BROKEN: Config object construction (internal API, not exported in some wheels)
config = sherpa_onnx.OnlineRecognizerConfig(...)  # AttributeError!
recognizer = sherpa_onnx.OnlineRecognizer(config)
```

**Current setup:**
- `sherpa-onnx==1.12.14+cuda12.cudnn9` (CUDA-enabled wheel with bundled ORT)
- Factory method API that works consistently across all wheel builds
- No more dependency on internal config classes
- **Important:** PyPI `sherpa-onnx` doesn't have GPU support compiled in - must use CUDA wheel!

## Quick start (Docker, GPU)

```bash
# Build
docker build -t sherpa-asr:gpu -f docker/Dockerfile .

# Run (all GPUs) - fastest decoding
docker run --rm --gpus all -p 8000:8000 \
  -e PROVIDER=cuda \
  -e MAX_BATCH=64 \
  -e PARTIAL_HZ=20 \
  --name sherpa-asr sherpa-asr:gpu

# Run with better quality (slower but improved transcription)
docker run --rm --gpus all -p 8000:8000 \
  -e PROVIDER=cuda \
  -e DECODING_METHOD=modified_beam_search \
  -e MAX_ACTIVE_PATHS=8 \
  --name sherpa-asr sherpa-asr:gpu
```

Server starts at `ws://0.0.0.0:8000/ws`.

## WebSocket protocol
- Send binary frames: PCM16LE mono at 16 kHz (any chunking; **10ms recommended for low latency**)
- Terminate stream with control frame: `b"__CTRL__:EOS"`
- Receive JSON text frames:
  - Partial: `{ "type": "partial", "text": "..." }` (no punctuation)
  - Final: `{ "type": "final", "text": "..." }` (stable segment text)

## Config (env vars)
- `WS_HOST` (default `0.0.0.0`)
- `WS_PORT` (default `8000`)
- `SAMPLE_RATE` (default `16000`)
- `PROVIDER` (`cuda` or `cpu`, default `cuda`)
- `NUM_THREADS` (ONNX Runtime threads, default `6`)
- `MAX_BATCH` (default `64`)
- `MAX_CONNECTIONS` (default `2048`)
- `PARTIAL_HZ` (partials per second per client; default `20`)
- `DECODING_METHOD` (`greedy_search` or `modified_beam_search`, default `greedy_search`)
- `MAX_ACTIVE_PATHS` (beam search width, default `8`)
- `DRAIN_BUDGET_MS` (decode time budget per loop; default `200`)
- `ENDPOINT_RULE1_MS`/`RULE2_MS`/`RULE3_MIN_UTT_MS` (default `800/400/800`)
- `ASR_DIR` (default `/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-21`)

## Included models
During the Docker build a separate builder stage:
- Clones the Icefall exporter and the HF checkpoint
- Runs `export-onnx.py` with `--decode-chunk-len 16`
- Produces chunked streaming ONNX files and copies them into `/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-21`

You can override `ASR_DIR` at runtime to point to a different model directory if you copy your own models into the container.

## Smoke tests (included clients)

### From host machine (external):
```bash
WS=ws://127.0.0.1:8000/ws python3 tests/client.py --file samples/mid.wav --print-partials --full-text
```

### From inside container:
```bash
# Warmup test (fast send to warm GPU kernels)
python3 tests/warmup.py --file realistic-2.mp3 --rtf 1.0 --full-text

# Concurrency benchmark (synthetic audio)
python3 tests/bench.py --streams 64 --duration 30 --frame-ms 20 --rtf 1.0 --print-partials

# Test different files
python3 tests/warmup.py --file realistic.mp3 --rtf 1.0 --print-partials --full-text
python3 tests/warmup.py --file long-noisy.mp3 --rtf 2.0 --full-text
```

The `samples/` and `tests/` directories are copied into the image and available at `/app/samples` and `/app/tests`.

## Troubleshooting

### GPU not working?
Check GPU support inside the container:
```bash
# Check sherpa-onnx factory method and version
docker exec -it sherpa-asr python - <<'PY'
import sherpa_onnx as so, inspect
print("sherpa-onnx version:", so.__version__)
print("has from_transducer:", hasattr(so.OnlineRecognizer, "from_transducer"))
print("CUDA wheel should have '+cuda12.cudnn9' in version")
PY

# Verify GPU is actually being used (check container logs)
docker logs sherpa-asr | grep -E "(Fallback to cpu|DSHERPA_ONNX_ENABLE_GPU)"
# Should NOT see: "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON" or "Fallback to cpu!"

# Check NVIDIA runtime is available
docker exec -it sherpa-asr nvidia-smi
docker exec -it sherpa-asr ls -l /dev/nvidia*
```

### sherpa-onnx API errors?
If you see `OnlineRecognizer() takes no arguments` or missing config classes:
1. **Check your sherpa-onnx version:** Different wheels export different APIs
2. **Use factory methods:** `OnlineRecognizer.from_transducer()` is the stable public API
3. **Avoid config objects:** They're internal and not exported in some wheel builds

### Common issues:
- **"missing chunked *.onnx":** You're using a model pack without chunked streaming exports (like 2023-06-21). Use 2023-06-21 or re-export with chunked streaming enabled
- **Poor real-time performance (RTF > 1.0):** Check logs for chunked file usage; should see "using encoder=...chunk..." in startup logs  
- **"Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON":** You're using PyPI sherpa-onnx (no GPU). Use the CUDA wheel instead: `sherpa-onnx==1.12.14+cuda12.cudnn9`
- **`import onnxruntime` fails:** CUDA wheel bundles ORT natively - no Python `onnxruntime` module available (this is expected!)
- **Import errors:** Make sure you're using the CUDA wheel, not mixing with `onnxruntime-gpu`
- **CUDA out of memory:** Lower `MAX_BATCH` or use smaller models
- **High latency:** Increase `PARTIAL_HZ`, tighten endpointing rules, or check GPU utilization
- **Poor transcription quality:** Try `DECODING_METHOD=modified_beam_search` with `MAX_ACTIVE_PATHS=8`
- **"Fallback to cpu!":** Check container logs and NVIDIA runtime with debugging commands above

## Why this stack
- **Sherpa-ONNX Zipformer RNNT:** optimized for streaming; exposes `decode_streams()` for batch GPU decode  
- **Factory method API:** stable public interface that works across all wheel builds
- **CUDA-enabled sherpa-onnx wheel:** includes GPU-compiled binaries + bundled ORT for reliable CUDA 12.x + cuDNN 9 support
- **Async WebSocket server:** handles thousands of concurrent streams with batched GPU processing

## Notes & tuning
- **Lower latency:** increase `PARTIAL_HZ` (e.g., 30–40) and/or tighten endpointing
- **Higher throughput:** raise `MAX_BATCH` if GPU has headroom  
- **Better quality:** use `DECODING_METHOD=modified_beam_search` with `MAX_ACTIVE_PATHS=8` (costs ~10% latency)
- **CPU fallback:** set `PROVIDER=cpu` (much slower)
- **Chunked streaming:** Server enforces chunked exports and will fail fast if missing

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
- **Sherpa-ONNX GPU install:** `https://k2-fsa.github.io/sherpa/onnx/python/install.html`
- **Streaming Zipformer (EN):** `https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html#streaming-zipformer-en-2023-06-21`
- **sherpa-onnx releases:** `https://github.com/k2-fsa/sherpa-onnx/releases`
