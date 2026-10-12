"""Download movie tars from Spaces on the droplet, several at once.

Reads "<collection> <movie> <part-url> [<part-url> ...]" lines on stdin (sent by
launch.py; no Spaces key on the droplet), streams each tar into `tar -x`
under /scratch/frames/<collection>/, and touches `.complete` the moment a
movie has arrived so run_index.py starts indexing it. Touches
/scratch/frames/.uploads-done when everything has arrived.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FRAMES = Path(__import__("os").environ.get("FRAMES", "/scratch/frames"))
PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 8

started = time.monotonic()
lock = threading.Lock()
stats = {"bytes": 0, "movies": 0, "failed": 0}


def say(message: str) -> None:
    print(f"[download {time.monotonic() - started:5.0f}s] {message}", flush=True)


def fetch_part(url: str, target: Path, label: str) -> bool:
    for attempt in range(1, 6):
        sink = subprocess.Popen(["tar", "-x", "-C", str(target)], stdin=subprocess.PIPE)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(4 << 20):
                    sink.stdin.write(chunk)
                    with lock:
                        stats["bytes"] += len(chunk)
            sink.stdin.close()
            if sink.wait() == 0:
                return True
        except Exception as error:
            say(f"{label} attempt {attempt} failed: {error}")
            sink.kill()
        time.sleep(2 ** attempt)
    return False


def fetch(line: str) -> None:
    collection, movie, *urls = line.split()
    target = FRAMES / collection
    target.mkdir(parents=True, exist_ok=True)
    for n, url in enumerate(urls):  # parts extract into the same <movie>/ folder
        if not fetch_part(url, target, f"{movie} part {n}"):
            with lock:
                stats["failed"] += 1
            say(f"giving up on {movie}")
            return
    (target / movie / ".complete").touch()
    with lock:
        stats["movies"] += 1
    say(f"arrived {movie} ({len(urls)} parts) -> GPU")


def report(total: int, stop: threading.Event) -> None:
    last, last_time = 0, time.monotonic()
    while not stop.wait(5):
        now = time.monotonic()
        with lock:
            got, movies = stats["bytes"], stats["movies"]
        say(f"{movies}/{total} movies  {got / 1e9:6.2f} GB  {(got - last) / (now - last_time) / 1e6:5.0f} MB/s")
        last, last_time = got, now


def main() -> None:
    lines = [line.strip() for line in sys.stdin if line.strip()]
    say(f"{len(lines)} movies, {PARALLEL} at a time")
    stop = threading.Event()
    threading.Thread(target=report, args=(len(lines), stop), daemon=True).start()
    with ThreadPoolExecutor(PARALLEL) as pool:
        list(pool.map(fetch, lines))
    stop.set()
    (FRAMES / ".uploads-done").touch()
    elapsed = time.monotonic() - started
    say(f"all arrived: {stats['movies']}/{len(lines)} movies, {stats['bytes'] / 1e9:.1f} GB in {elapsed:.0f}s "
        f"({stats['bytes'] / elapsed / 1e6:.0f} MB/s avg), {stats['failed']} failed")


if __name__ == "__main__":
    main()
