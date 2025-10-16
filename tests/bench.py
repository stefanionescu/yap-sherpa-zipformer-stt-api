#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from utils import SAMPLE_RATE, load_audio, resolve_sample_path, stream_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real audio concurrency bench for the ASR server")
    parser.add_argument("--url", default=os.getenv("WS", "ws://127.0.0.1:8000"), help="WebSocket endpoint")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--streams", type=int, default=32, help="Number of concurrent streams")
    parser.add_argument("--frame-ms", type=int, default=10, help="Frame size in milliseconds (10ms optimized for low latency)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor for sending audio")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    # Load real audio file like warmup.py does
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return
    audio = load_audio(audio_path)
    # Ensure contiguous int16 for consistent wire format (s16le)
    if audio.dtype != "int16":
        audio = np.clip(audio.astype("float32", copy=False), -1.0, 1.0)
        audio = (audio * 32767.0).astype("int16")
    else:
        audio = audio.astype("int16", copy=False)
    
    duration = len(audio) / SAMPLE_RATE

    async def session() -> None:
        await stream_session(
            args.url,
            audio,
            frame_ms=args.frame_ms,
            rtf=args.rtf,
            print_partials=args.print_partials,
        )

    print(f"Starting {args.streams} concurrent streams with {audio_path.name} ({duration:.1f}s)")
    
    tasks = [asyncio.create_task(session()) for _ in range(int(args.streams))]
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    print(f"Completed: streams={args.streams} audio_duration={duration:.1f}s wall_time={elapsed:.2f}s")
    print(f"Throughput: {args.streams * duration / elapsed:.1f}x realtime")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
