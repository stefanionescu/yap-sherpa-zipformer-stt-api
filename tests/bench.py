#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from utils import SAMPLE_RATE, stream_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic concurrency bench for the Vosk server")
    parser.add_argument("--url", default=os.getenv("WS", "ws://127.0.0.1:8000"), help="WebSocket endpoint")
    parser.add_argument("--streams", type=int, default=32, help="Number of concurrent streams")
    parser.add_argument("--duration", type=int, default=30, help="Synthetic duration in seconds per stream")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor for sending audio")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses")
    return parser.parse_args()


def synth(duration: int) -> np.ndarray:
    t = np.arange(0, duration * SAMPLE_RATE) / SAMPLE_RATE
    tone = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.12 * np.sin(2 * np.pi * 310 * t)
    return (tone * 32767.0).astype(np.int16)


async def run(args: argparse.Namespace) -> None:
    audio = synth(args.duration).astype(np.int16, copy=False)

    async def session() -> None:
        await stream_session(
            args.url,
            audio,
            frame_ms=args.frame_ms,
            rtf=args.rtf,
            print_partials=args.print_partials,
        )

    tasks = [asyncio.create_task(session()) for _ in range(int(args.streams))]
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    print(f"Streams={args.streams} wall={elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
