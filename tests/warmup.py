#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from utils import SAMPLE_RATE, load_audio, resolve_sample_path, stream_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup Vosk streaming endpoint")
    parser.add_argument("--url", default=os.getenv("WS", "ws://127.0.0.1:8000"), help="WebSocket endpoint")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (higher = faster send)")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--print-partials", action="store_true", help="Emit partial hypotheses")
    parser.add_argument("--full-text", action="store_true", help="Print the full final transcript")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    audio = load_audio(audio_path)
    # Ensure contiguous int16 for consistent wire format (s16le)
    if audio.dtype != "int16":
        import numpy as np  # local import to avoid global dependency here
        audio = np.clip(audio.astype("float32", copy=False), -1.0, 1.0)
        audio = (audio * 32767.0).astype("int16")
    else:
        audio = audio.astype("int16", copy=False)
    duration = len(audio) / SAMPLE_RATE
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
    ttfw = result.get("ttfw_s")
    print(f"Audio duration: {duration:.3f}s")
    print(f"Wall-to-final: {wall:.3f}s")
    if duration > 0:
        print(f"RTF(measured): {wall / duration:.3f}  xRT: {duration / wall if wall > 0 else 0.0:.2f}x")
    print(f"Partials: {int(result.get('partials', 0))}")
    if isinstance(ttfw, (int, float)):
        print(f"TTFW: {ttfw * 1000.0:.1f} ms")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())
