#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 16_000
CONTROL_PREFIX = b"__CTRL__:"
CTRL_EOS = b"EOS"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def resolve_sample_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.parts and path.parts[0] == "samples":
        path = Path(*path.parts[1:])
    return SAMPLES_DIR / path


def _ffmpeg_decode_to_pcm16_mono_16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.int16)


def load_audio(path: Path) -> np.ndarray:
    try:
        audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype("int16")
        if sr != SAMPLE_RATE:
            return _ffmpeg_decode_to_pcm16_mono_16k(path)
        return np.asarray(audio, dtype=np.int16)
    except Exception:
        return _ffmpeg_decode_to_pcm16_mono_16k(path)


def _chunk_audio(audio: np.ndarray, samples_per_chunk: int) -> Iterable[np.ndarray]:
    total = len(audio)
    for start in range(0, total, samples_per_chunk):
        yield audio[start : start + samples_per_chunk]


def _to_s16le_bytes(frame: np.ndarray) -> bytes:
    """Convert a numpy audio frame (int16 or float) to little-endian int16 bytes."""
    if frame.dtype == np.int16:
        return frame.astype("<i2", copy=False).tobytes()
    # Assume float-like [-1, 1]
    f = np.clip(frame.astype(np.float32, copy=False), -1.0, 1.0)
    return (f * 32767.0).astype("<i2").tobytes()


async def stream_session(
    url: str,
    audio: np.ndarray,
    *,
    frame_ms: int = 20,
    rtf: float = 1.0,
    print_partials: bool = False,
) -> Dict[str, Any]:
    samples_per_chunk = max(1, int(SAMPLE_RATE * (frame_ms / 1000.0)))
    start_ts = time.perf_counter()
    first_audio_ts: float | None = None
    first_partial_latency: float | None = None
    final_ts: float | None = None
    partial_count = 0
    final_text = ""
    done_event = asyncio.Event()

    async with websockets.connect(url, max_size=2**23) as ws:
        async def sender() -> None:
            nonlocal first_audio_ts
            sent_samples = 0
            send_start = time.perf_counter()
            for frame in _chunk_audio(audio, samples_per_chunk):
                if frame.size == 0:
                    continue
                await ws.send(_to_s16le_bytes(frame))
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter()
                sent_samples += int(frame.size)
                target = send_start + (sent_samples / SAMPLE_RATE) / max(rtf, 1e-6)
                sleep_for = target - time.perf_counter()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)
            await ws.send(CONTROL_PREFIX + CTRL_EOS)
            try:
                await asyncio.wait_for(done_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

        async def receiver() -> None:
            nonlocal final_text, final_ts, partial_count, first_partial_latency
            async for message in ws:
                now = time.perf_counter()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type") or "")
                text = str(payload.get("text") or "")
                if msg_type == "partial":
                    partial_count += 1
                    if print_partials and text:
                        print(f"[partial] {text}")
                    if first_audio_ts is not None and first_partial_latency is None:
                        first_partial_latency = now - first_audio_ts
                elif msg_type == "final":
                    final_text = text
                    final_ts = now
                    if print_partials and text:
                        print(f"[final] {text}")
                    done_event.set()
                    return
            done_event.set()

        await asyncio.gather(sender(), receiver())
        if not ws.closed:
            await ws.close()

    end_ts = final_ts or time.perf_counter()
    audio_s = len(audio) / SAMPLE_RATE
    wall_s = end_ts - start_ts
    return {
        "text": final_text,
        "audio_s": audio_s,
        "wall_s": wall_s,
        "rtf": (wall_s / audio_s) if audio_s > 0 else float("inf"),
        "xrt": (audio_s / wall_s) if wall_s > 0 else 0.0,
        "partials": partial_count,
        "ttfw_s": first_partial_latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vosk streaming smoke test")
    parser.add_argument("--url", default=os.getenv("WS", "ws://127.0.0.1:8000"), help="WebSocket endpoint")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0 = real-time)")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses")
    parser.add_argument("--full-text", action="store_true", help="Print full final transcript")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    audio = load_audio(audio_path)
    result = await stream_session(
        args.url,
        audio,
        frame_ms=args.frame_ms,
        rtf=args.rtf,
        print_partials=args.print_partials,
    )
    text = str(result.get("text", ""))
    if args.full_text:
        print(f"Final: {text}")
    else:
        print(f"Final: {text[:80]}…" if len(text) > 80 else f"Final: {text}")
    wall = float(result.get("wall_s", 0.0))
    audio_s = float(result.get("audio_s", 0.0))
    rtf = float(result.get("rtf", 0.0))
    xrt = float(result.get("xrt", 0.0))
    ttfw = result.get("ttfw_s")
    print(
        "Wall={:.3f}s  Audio={:.3f}s  RTF={:.3f}  xRT={:.2f}x  Partials={}  TTFW={:.1f}ms".format(
            wall,
            audio_s,
            rtf,
            xrt,
            int(result.get("partials", 0)),
            (ttfw * 1000.0) if isinstance(ttfw, (int, float)) else float("nan"),
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))




if __name__ == "__main__":
    raise SystemExit(main())
