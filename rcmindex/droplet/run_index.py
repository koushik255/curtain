"""Index movies on the GPU as they finish uploading.

Runs on the droplet. launch.py uploads movie folders into
/scratch/frames/<collection>/<movie>/ and touches `.complete` in each one when
it has fully arrived, then touches /scratch/frames/.uploads-done at the end.
This script hands every completed folder to the already-running rcmindex
immediately, so the GPU works while later movies are still uploading.

    python3 run_index.py                    # index everything that arrives
    python3 run_index.py -- --limit 4096    # extra rcmindex arguments after --
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RCMINDEX = Path(os.environ.get("RCMINDEX", HERE.parent / "build/rcmindex"))

started = time.monotonic()
lock = threading.Lock()
stats = {"queued": 0, "indexed": 0, "frames": 0, "gpu_seconds": 0.0}


def say(message: str) -> None:
    print(f"[gpu {time.monotonic() - started:5.0f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("/scratch/frames"))
    parser.add_argument("--output", type=Path, default=Path("/scratch/out"))
    parser.add_argument("--model", type=Path, default=Path.home() / "rcmindex-model")
    parser.add_argument("--delete-frames", action="store_true")
    parser.add_argument("rcmindex_args", nargs="*")
    args = parser.parse_args()
    done_marker = args.frames / ".uploads-done"

    say("starting rcmindex (first run compiles the model, a few minutes)")
    gpu = subprocess.Popen(
        [str(RCMINDEX), "--model", str(args.model), "--movies-from", "-", "--output", str(args.output),
         "--global", *args.rcmindex_args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )

    def on_done() -> None:
        # rcmindex prints "DONE <dir> <frames> <fps>" per movie.
        for line in gpu.stdout:
            parts = line.split()
            if len(parts) != 4 or parts[0] != "DONE":
                continue
            directory, frames, fps = Path(parts[1]), int(parts[2]), float(parts[3])
            with lock:
                stats["indexed"] += 1
                stats["frames"] += frames
                stats["gpu_seconds"] += frames / fps if fps else 0
                s = dict(stats)
            say(f"indexed {directory.name}: {frames:,} frames at {fps:,.0f} fr/s "
                f"({s['indexed']}/{s['queued']} movies, {s['frames']:,} frames so far)")
            if args.delete_frames:
                subprocess.run(["rm", "-rf", str(directory)])

    reader = threading.Thread(target=on_done)
    reader.start()

    seen: set[Path] = set()
    while True:
        finished = done_marker.exists()  # check before scanning so nothing is missed
        for marker in sorted(args.frames.glob("*/*/.complete")):
            directory = marker.parent
            if directory in seen:
                continue
            seen.add(directory)
            with lock:
                stats["queued"] += 1
            say(f"queued {directory.name}")
            try:
                gpu.stdin.write(f"{directory}\n")
                gpu.stdin.flush()
            except BrokenPipeError:
                reader.join()
                sys.exit(f"rcmindex crashed (exit code {gpu.wait()}); see its error above")
        if finished:
            break
        if gpu.poll() is not None:  # crashed while waiting for the next upload
            reader.join()
            sys.exit(f"rcmindex crashed (exit code {gpu.returncode}); see its error above")
        time.sleep(1)

    gpu.stdin.close()
    code = gpu.wait()
    reader.join()
    s = stats
    fps = s["frames"] / s["gpu_seconds"] if s["gpu_seconds"] else 0
    say(f"finished: {s['indexed']}/{s['queued']} movies, {s['frames']:,} frames, "
        f"{fps:,.0f} fr/s average on GPU, {time.monotonic() - started:.0f}s total")
    sys.exit(code)


if __name__ == "__main__":
    main()
