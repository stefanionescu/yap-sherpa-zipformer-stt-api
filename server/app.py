#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import websockets
import websockets.server
import uvloop  # type: ignore
import sherpa_onnx as so

# Quiet runtime settings for performance
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")  # 3=error (quiet)
os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")
os.environ.setdefault("SHERPA_ONNX_LOG", "0")

print(f"[sherpa-asr] sherpa_onnx version: {so.__version__}")

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
LOG = logging.getLogger("sherpa-asr")

# ---- Protocol constants (match bench/client) ----
CONTROL_PREFIX = b"__CTRL__:"
CTRL_SEG = b"SEG"  # commit current segment, keep connection open
CTRL_EOS = b"EOS"  # finalize segment and end session
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# ---- Runtime tuning ----
HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8000"))
PROVIDER = os.getenv("PROVIDER", "cuda")  # "cuda" or "cpu"
MAX_BATCH = max(1, int(os.getenv("MAX_BATCH", "64")))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "2048"))
PARTIAL_HZ = float(os.getenv("PARTIAL_HZ", "60"))  # partials per second per client (increased for snappier partials)
EMIT_INTERVAL = 1.0 / max(1.0, PARTIAL_HZ)

# Decode scheduling knobs (optimized for throughput)
DRAIN_BUDGET_MS = float(os.getenv("DRAIN_BUDGET_MS", "200"))  # generous for single-stream baseline
LOOP_SLEEP = float(os.getenv("LOOP_SLEEP", "0.0005"))

# Micro-batching knobs
MAX_WAIT_MS = float(os.getenv("MAX_WAIT_MS", "8"))          # tiny micro-batch window
MIN_BATCH_BEFORE_WAIT = int(os.getenv("MIN_BATCH_BEFORE_WAIT", "4"))  # if we already have 4+, don't wait

# Endpointing (aggressive for low latency)
ENDPOINT_RULE1_MS = int(os.getenv("ENDPOINT_RULE1_MS", "800"))
ENDPOINT_RULE2_MS = int(os.getenv("ENDPOINT_RULE2_MS", "400"))
ENDPOINT_RULE3_MIN_UTT_MS = int(os.getenv("ENDPOINT_RULE3_MIN_UTT_MS", "800"))

# Decoding method options
DECODING_METHOD = os.getenv("DECODING_METHOD", "greedy_search")  # or "modified_beam_search"
MAX_ACTIVE_PATHS = int(os.getenv("MAX_ACTIVE_PATHS", "8"))  # for beam search

ASR_DIR = Path(os.getenv("ASR_DIR", ""))


def _require_chunked(dir_: Path):
    """Fail fast if ASR_DIR lacks chunked streaming exports."""
    needed = ["encoder", "decoder", "joiner"]
    for role in needed:
        if not any("chunk" in p.name for p in dir_.glob(f"{role}*.onnx")):
            raise RuntimeError(
                f"{dir_} has no chunked {role}*.onnx. "
                "Use 2023-06-26 or re-export 2023-06-21 as chunked."
            )


def _pick_chunked(dir_: Path, role: str) -> str:
    """Pick the best chunked ONNX model file."""
    files = list(dir_.glob(f"{role}-epoch-99-avg-1-chunk-*.onnx"))
    
    if not files:
        raise RuntimeError(f"{dir_}: missing chunked {role}*.onnx; use the 2023-06-26 pack.")
    
    # Prefer left-128 fp32 on GPU, then left-64, then other chunks
    # Sort by: (is_int8, not_left_128, not_left_64, name)
    def score(p: Path) -> tuple[int, int, int, str]:
        name = p.name
        is_int8 = 1 if ".int8." in name else 0
        not_left_128 = 0 if "left-128" in name else 1
        not_left_64 = 0 if "left-64" in name else 1
        return (is_int8, not_left_128, not_left_64, name)
    
    files.sort(key=score)
    pick = files[0]
    LOG.info("using %s=%s", role, pick.name)
    return str(pick)


def _load_asr() -> so.OnlineRecognizer:
    """Load ASR model using the canonical sherpa-onnx public API (factory method)."""
    # Ensure we have chunked streaming exports
    _require_chunked(ASR_DIR)
    
    # Pick the best chunked models
    enc = _pick_chunked(ASR_DIR, "encoder")
    dec = _pick_chunked(ASR_DIR, "decoder")
    joi = _pick_chunked(ASR_DIR, "joiner")
    tok = str(ASR_DIR / "tokens.txt")

    NUM_THREADS = int(os.getenv("NUM_THREADS", "6"))  # good start for 28 vCPUs

    # sherpa-onnx canonical public API (works in 1.12.13 and 1.12.14):
    try:
        recognizer = so.OnlineRecognizer.from_transducer(
            tokens=tok,
            encoder=enc,
            decoder=dec,
            joiner=joi,
            # runtime / features
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            # endpointing (seconds) - disabled for client-driven control
            enable_endpoint_detection=False,   # client controls end via EOS
            rule1_min_trailing_silence=ENDPOINT_RULE1_MS / 1000.0,
            rule2_min_trailing_silence=ENDPOINT_RULE2_MS / 1000.0,
            rule3_min_utterance_length=ENDPOINT_RULE3_MIN_UTT_MS / 1000.0,
            # decoding
            decoding_method=DECODING_METHOD,
            max_active_paths=MAX_ACTIVE_PATHS,
            # provider
            provider=("cuda" if PROVIDER == "cuda" else "cpu"),
            device=0
        )
        
        # Hard-fail if CUDA was requested but not available
        if PROVIDER == "cuda":
            LOG.info("✅ CUDA provider successfully initialized")
        
        LOG.info("num_threads(features)=%d", NUM_THREADS)
        return recognizer
        
    except Exception as e:
        if PROVIDER == "cuda":
            raise SystemExit(f"FATAL: CUDA provider not available: {e}")
        else:
            raise


RECOGNIZER: Optional[so.OnlineRecognizer] = None


def _need_recognizer() -> so.OnlineRecognizer:
    """Guard to ensure recognizer is initialized before use."""
    if RECOGNIZER is None:
        raise RuntimeError("Recognizer not initialized")
    return RECOGNIZER


def _warmup():
    """One-time GPU warm-up to reduce first-partial latency."""
    r = _need_recognizer()
    s = r.create_stream()
    s.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    while r.is_ready(s):
        r.decode_stream(s)
    r.reset(s)


async def _decode_group_with_micro_wait(recog, clients, budget_deadline: float) -> bool:
    """
    Decode 'clients' whose streams are ready. If too few are ready, wait up to MAX_WAIT_MS
    (bounded by remaining budget) to accumulate a larger batch. Returns True if any progress.
    """
    if not clients:
        return False

    def ready_streams():
        return [c.stream for c in clients if recog.is_ready(c.stream)]

    ready = ready_streams()
    if not ready:
        return False

    # If batch is small, briefly wait for more (but never exceed loop budget)
    if len(ready) < MIN_BATCH_BEFORE_WAIT and MAX_WAIT_MS > 0.0:
        # how much time left in the outer budget?
        left_ms = max(0.0, (budget_deadline - time.perf_counter()) * 1000.0)
        wait_ms = min(MAX_WAIT_MS, left_ms)
        t_end = time.perf_counter() + wait_ms / 1000.0
        while time.perf_counter() < t_end:
            await asyncio.sleep(0)  # yield so other tasks can feed audio
            r2 = ready_streams()
            if len(r2) >= MIN_BATCH_BEFORE_WAIT or len(r2) >= MAX_BATCH:
                ready = r2
                break
            ready = r2  # keep freshest view

    # Batch in chunks of MAX_BATCH
    progressed = False
    for i in range(0, len(ready), MAX_BATCH):
        batch = ready[i:i+MAX_BATCH]
        if not batch:
            continue
        recog.decode_streams(batch)
        progressed = True
    return progressed


@dataclass
class Client:
    ws: websockets.server.WebSocketServerProtocol
    stream: "so.OnlineStream"
    last_partial: str = ""
    last_emit_ts: float = field(default_factory=time.perf_counter)
    ended: bool = False
    commit: bool = False
    done: bool = False


CLIENTS: Dict[int, Client] = {}
_CLIENT_SEQ = 0
_CLIENTS_LOCK = asyncio.Lock()


def _next_id() -> int:
    global _CLIENT_SEQ
    _CLIENT_SEQ += 1
    return _CLIENT_SEQ


def _is_conn_closed(ws) -> bool:
    """Check if websocket connection is closed (compatible with websockets 12).
    
    websockets 12 ServerConnection has no `.closed` property.
    Close is observable via close_code or state.
    """
    cc = getattr(ws, "close_code", None)
    if cc is not None:
        return True
    state = getattr(ws, "state", None)  # enum in websockets 12
    name = getattr(state, "name", "")
    return name in {"CLOSING", "CLOSED"}


async def decode_loop():
    while True:
        await asyncio.sleep(LOOP_SLEEP)

        recog = _need_recognizer()

        # 1) Drain decoding within time budget, prioritizing finalizing ('hot') streams
        budget_deadline = time.perf_counter() + (DRAIN_BUDGET_MS / 1000.0)
        while time.perf_counter() < budget_deadline:
            # split into hot/cold each iteration so priority is fresh
            hot_clients  = [c for c in CLIENTS.values() if (c.commit or c.ended)]
            cold_clients = [c for c in CLIENTS.values() if not (c.commit or c.ended)]

            progressed = False
            # Hot first: let finals/commits jump the queue
            if await _decode_group_with_micro_wait(recog, hot_clients, budget_deadline):
                progressed = True
            if await _decode_group_with_micro_wait(recog, cold_clients, budget_deadline):
                progressed = True

            if not progressed:
                break

        # 2) Emit partials + handle finals
        now = time.perf_counter()
        for cid, c in list(CLIENTS.items()):
            try:
                # Partials
                if (now - c.last_emit_ts) >= EMIT_INTERVAL:
                    text = recog.get_result(c.stream).strip()
                    if text and text != c.last_partial:
                        await safe_send(c.ws, {"type": "partial", "text": text})
                        c.last_partial = text
                    c.last_emit_ts = now

                # Finals: EOS/SEG => drain to completion (NO time cap)
                if c.commit or c.ended:
                    while recog.is_ready(c.stream):
                        recog.decode_stream(c.stream)
                    final_text = recog.get_result(c.stream).strip()
                    await safe_send(c.ws, {"type": "final", "text": final_text})
                    recog.reset(c.stream)
                    c.commit = False
                    if c.ended:
                        c.done = True

            except Exception:
                traceback.print_exc()
                c.done = True

        # 3) Cleanup
        to_close = [cid for cid, c in CLIENTS.items() if c.done or _is_conn_closed(c.ws)]
        for cid in to_close:
            try:
                CLIENTS[cid].stream = None  # type: ignore
            finally:
                CLIENTS.pop(cid, None)


async def safe_send(ws: websockets.server.WebSocketServerProtocol, obj: dict):
    # Compatible with websockets 12
    if getattr(ws, "close_code", None) is not None:
        return
    state = getattr(ws, "state", None)
    if getattr(state, "name", "") in {"CLOSING", "CLOSED"}:
        return
    try:
        await ws.send(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass


async def handle_ws(ws: websockets.server.WebSocketServerProtocol):
    async with _CLIENTS_LOCK:
        if len(CLIENTS) >= MAX_CONNECTIONS:
            await ws.close(code=1013, reason="Overloaded")
            return
        stream = _need_recognizer().create_stream()
        cid = _next_id()
        CLIENTS[cid] = Client(ws=ws, stream=stream)

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                b = bytes(msg)
                if b.startswith(CONTROL_PREFIX):
                    ctrl = b[len(CONTROL_PREFIX) :]
                    if ctrl == CTRL_SEG:
                        # Check if client still exists (might be removed by decode_loop)
                        if cid in CLIENTS:
                            CLIENTS[cid].commit = True
                    elif ctrl == CTRL_EOS:
                        # Check if client still exists (might be removed by decode_loop)
                        if cid in CLIENTS:
                            CLIENTS[cid].ended = True
                            CLIENTS[cid].stream.input_finished()
                    continue
                if len(b) % 2 != 0:
                    b = b[: len(b) - 1]
                if not b:
                    continue
                pcm = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
                # Check if client still exists (might be removed by decode_loop)
                if cid in CLIENTS:
                    CLIENTS[cid].stream.accept_waveform(SAMPLE_RATE, pcm)
            else:
                continue
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        if cid in CLIENTS:
            CLIENTS[cid].done = True


async def main():
    global RECOGNIZER
    
    # Runtime diagnostics - prove what's actually loaded
    LOG.info("🔍 Runtime Diagnostics:")
    LOG.info("sherpa_onnx=%s  file=%s", so.__version__, so.__file__)
    
    # Show ORT library candidates
    root = os.path.dirname(so.__file__)
    ort_libs = glob.glob(os.path.join(root, "**/libonnxruntime.*"), recursive=True)
    LOG.info("libonnxruntime candidates: %s", ort_libs)
    
    # CUDA environment
    LOG.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    LOG.info("NVIDIA_VISIBLE_DEVICES=%s", os.environ.get("NVIDIA_VISIBLE_DEVICES", "not set"))
    
    # ORT logging settings
    LOG.info("ORT_LOG_SEVERITY_LEVEL=%s ORT_LOG_VERBOSITY_LEVEL=%s SHERPA_ONNX_LOG=%s", 
             os.environ.get("ORT_LOG_SEVERITY_LEVEL"), 
             os.environ.get("ORT_LOG_VERBOSITY_LEVEL"),
             os.environ.get("SHERPA_ONNX_LOG"))
    
    # Preflight logging - show configuration before attempting to load
    LOG.info("ASR_DIR=%s PROVIDER=%s SAMPLE_RATE=%s", ASR_DIR, PROVIDER, SAMPLE_RATE)
    LOG.info("EMIT_INTERVAL=%.3f (PARTIAL_HZ=%.1f) MAX_BATCH=%d", EMIT_INTERVAL, PARTIAL_HZ, MAX_BATCH)
    LOG.info("DRAIN_BUDGET_MS=%.1f (optimized for throughput)", DRAIN_BUDGET_MS)
    LOG.info("MAX_CONNECTIONS=%d HOST=%s PORT=%s", MAX_CONNECTIONS, HOST, PORT)
    LOG.info("Decoding: method=%s max_active_paths=%d", DECODING_METHOD, MAX_ACTIVE_PATHS)
    LOG.info("Endpoint rules: RULE1=%dms RULE2=%dms RULE3_MIN_UTT=%dms", 
             ENDPOINT_RULE1_MS, ENDPOINT_RULE2_MS, ENDPOINT_RULE3_MIN_UTT_MS)
    
    try:
        LOG.info("Loading ASR model...")
        if PROVIDER == "cuda":
            LOG.info("🔍 Watch for 'CUDAExecutionProvider' in ORT logs below to confirm GPU usage...")
        RECOGNIZER = _load_asr()
        LOG.info("✅ Zipformer loaded (provider=%s) from %s", PROVIDER, ASR_DIR)
        
        LOG.info("Warming up GPU...")
        _warmup()
        LOG.info("✅ GPU warmup complete")
    except Exception:
        LOG.exception("FATAL: failed to load recognizer")
        # Keep container alive so logs are visible
        LOG.error("Container will sleep for 1 hour to allow log inspection...")
        await asyncio.sleep(3600)
        return

    asyncio.create_task(decode_loop())

    async with websockets.serve(
        handle_ws,
        host=HOST,
        port=PORT,
        max_size=2**23,
        ping_interval=None,   # disable keepalive for local testing
        ping_timeout=None,
        compression=None,
        max_queue=1024,
    ):
        LOG.info("🚀 WebSocket ready on ws://%s:%s  provider=%s  batch=%s", HOST, PORT, PROVIDER, MAX_BATCH)
        stop = asyncio.Future()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, stop.cancel)
        try:
            await stop
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())


