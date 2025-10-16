#!/usr/bin/env python3
from __future__ import annotations

import asyncio
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
CTRL_EOS = b"EOS"
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# ---- Runtime tuning ----
HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8000"))
PROVIDER = os.getenv("PROVIDER", "cuda")  # "cuda" or "cpu"
MAX_BATCH = max(1, int(os.getenv("MAX_BATCH", "64")))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "2048"))
PARTIAL_HZ = float(os.getenv("PARTIAL_HZ", "20"))  # partials per second per client
EMIT_INTERVAL = 1.0 / max(1.0, PARTIAL_HZ)

# Endpointing (aggressive for low latency)
ENDPOINT_RULE1_MS = int(os.getenv("ENDPOINT_RULE1_MS", "800"))
ENDPOINT_RULE2_MS = int(os.getenv("ENDPOINT_RULE2_MS", "400"))
ENDPOINT_RULE3_MIN_UTT_MS = int(os.getenv("ENDPOINT_RULE3_MIN_UTT_MS", "800"))

ASR_DIR = Path(os.getenv("ASR_DIR", ""))


def _pick_one(dir_: Path, prefix: str) -> str:
    """Return a model file path under dir_ with the given prefix, preferring chunked variants."""
    cand = sorted(p for p in dir_.glob(f"{prefix}*.onnx"))
    if not cand:
        raise FileNotFoundError(f"Missing {prefix}*.onnx in {dir_}")
    cand.sort(key=lambda p: (("chunk" not in p.name), p.name))
    return str(cand[0])


def _load_asr() -> so.OnlineRecognizer:
    """Load ASR model using the canonical sherpa-onnx public API (factory method)."""
    enc = _pick_one(ASR_DIR, "encoder")
    dec = _pick_one(ASR_DIR, "decoder")
    joi = _pick_one(ASR_DIR, "joiner")
    tok = str(ASR_DIR / "tokens.txt")

    # sherpa-onnx canonical public API (works in 1.12.13 and 1.12.14):
    return so.OnlineRecognizer.from_transducer(
        tokens=tok,
        encoder=enc,
        decoder=dec,
        joiner=joi,
        # runtime / features
        num_threads=max(1, int(os.getenv("NUM_THREADS", "2"))),
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        # endpointing (seconds)
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=ENDPOINT_RULE1_MS / 1000.0,
        rule2_min_trailing_silence=ENDPOINT_RULE2_MS / 1000.0,
        rule3_min_utterance_length=ENDPOINT_RULE3_MIN_UTT_MS / 1000.0,
        # decoding
        decoding_method="greedy_search",
        max_active_paths=4,
        # provider
        provider=("cuda" if PROVIDER == "cuda" else "cpu"),
        device=0,
        # you can tune these if needed:
        # cudnn_conv_algo_search=1,
        # debug=False,
    )


RECOGNIZER: Optional[so.OnlineRecognizer] = None


def _need_recognizer() -> so.OnlineRecognizer:
    """Guard to ensure recognizer is initialized before use."""
    if RECOGNIZER is None:
        raise RuntimeError("Recognizer not initialized")
    return RECOGNIZER


@dataclass
class Client:
    ws: websockets.server.WebSocketServerProtocol
    stream: "so.OnlineStream"
    last_partial: str = ""
    last_emit_ts: float = field(default_factory=time.perf_counter)
    ended: bool = False
    done: bool = False


CLIENTS: Dict[int, Client] = {}
_CLIENT_SEQ = 0
_CLIENTS_LOCK = asyncio.Lock()


def _next_id() -> int:
    global _CLIENT_SEQ
    _CLIENT_SEQ += 1
    return _CLIENT_SEQ


async def decode_loop():
    while True:
        await asyncio.sleep(0)
        ready: List[Client] = []
        for c in list(CLIENTS.values()):
            try:
                if _need_recognizer().is_ready(c.stream):
                    ready.append(c)
            except Exception:
                continue

        for i in range(0, len(ready), MAX_BATCH):
            part = ready[i : i + MAX_BATCH]
            if part:
                _need_recognizer().decode_streams([c.stream for c in part])

        now = time.perf_counter()
        for c in list(CLIENTS.values()):
            try:
                if (now - c.last_emit_ts) >= EMIT_INTERVAL:
                    res = _need_recognizer().get_result(c.stream)
                    text = (res.text or "").strip()
                    if text and text != c.last_partial:
                        await safe_send(c.ws, {"type": "partial", "text": text})
                        c.last_partial = text
                        c.last_emit_ts = now

                if _need_recognizer().is_endpoint(c.stream) or (c.ended and not _need_recognizer().is_ready(c.stream)):
                    final_res = _need_recognizer().get_result(c.stream)
                    final_text = (final_res.text or "").strip()
                    await safe_send(c.ws, {"type": "final", "text": final_text})
                    if c.ended:
                        c.done = True
                    _need_recognizer().reset(c.stream)
            except Exception:
                traceback.print_exc()
                c.done = True

        to_close = [cid for cid, c in CLIENTS.items() if c.done or c.ws.closed]
        for cid in to_close:
            try:
                CLIENTS[cid].stream = None  # type: ignore
            finally:
                CLIENTS.pop(cid, None)


async def safe_send(ws: websockets.server.WebSocketServerProtocol, obj: dict):
    if ws.closed:
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
                    if ctrl == CTRL_EOS:
                        CLIENTS[cid].ended = True
                        CLIENTS[cid].stream.input_finished()
                    continue
                if len(b) % 2 != 0:
                    b = b[: len(b) - 1]
                if not b:
                    continue
                pcm = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
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
    
    # Preflight logging - show configuration before attempting to load
    LOG.info("ASR_DIR=%s PROVIDER=%s SAMPLE_RATE=%s", ASR_DIR, PROVIDER, SAMPLE_RATE)
    LOG.info("EMIT_INTERVAL=%.3f MAX_BATCH=%d", EMIT_INTERVAL, MAX_BATCH)
    LOG.info("MAX_CONNECTIONS=%d HOST=%s PORT=%s", MAX_CONNECTIONS, HOST, PORT)
    LOG.info("Endpoint rules: RULE1=%dms RULE2=%dms RULE3_MIN_UTT=%dms", 
             ENDPOINT_RULE1_MS, ENDPOINT_RULE2_MS, ENDPOINT_RULE3_MIN_UTT_MS)
    
    try:
        # Debug: show what this wheel exports (helps future upgrades)
        names = [n for n in dir(so) if "Config" in n or "Recognizer" in n or "Stream" in n]
        LOG.info("sherpa_onnx %s exports: %s", getattr(so, "__version__", "?"), names)
        
        LOG.info("Loading ASR model...")
        RECOGNIZER = _load_asr()
        LOG.info("✅ Zipformer loaded (provider=%s) from %s", PROVIDER, ASR_DIR)
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
        ping_interval=20.0,
        ping_timeout=20.0,
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


