"""Upload every local movie to DigitalOcean Spaces once, in the background.

Each movie is split into parts of PART_FRAMES frames (~100 MB):

    frames/<collection>/<movie>/part-0000.tar ... part-NNNN.tar
    frames/<collection>/<movie>/manifest.json      written last

A part is one small atomic upload, retried on failure, so a dropped
connection costs at most one part. Parts already in the bucket are skipped,
so it can be stopped (Ctrl-C) and rerun any time and it picks up where it
left off. A movie counts as stored only once its manifest exists.

    python3 rcmindex/droplet/upload_frames.py               # everything (~114 GB)
    python3 rcmindex/droplet/upload_frames.py --movies 5    # next 5 unfinished movies
    python3 rcmindex/droplet/upload_frames.py --status
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spaces  # noqa: E402

spaces.ensure_boto3()

PART_FRAMES = 2000
started = time.monotonic()
lock = threading.Lock()
progress = {"sent": 0, "parts": 0, "movies": 0, "skipped_parts": 0}


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def frames_of(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == ".jpg")


def pack(movie: str, frames: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for frame in frames:
            tar.add(frame, arcname=f"{movie}/{frame.name}")
    return buffer.getvalue()


def put_part(s3, key: str, body: bytes) -> None:
    for attempt in range(1, 9):  # backs off up to ~4 minutes for a flaky connection
        try:
            s3.put_object(Bucket=spaces.BUCKET, Key=key, Body=body)
            return
        except Exception as error:
            wait = min(2 ** attempt, 60)
            say(f"retry {key.rsplit('/', 2)[-2]}/{key.rsplit('/', 1)[-1]} in {wait}s ({attempt}/8): {error}")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {key}")


def upload_movie(s3, pool: ThreadPoolExecutor, collection: str, directory: Path, existing: set[str]) -> None:
    movie = directory.name
    frames = frames_of(directory)
    chunks = [frames[i:i + PART_FRAMES] for i in range(0, len(frames), PART_FRAMES)]
    base = f"{spaces.PREFIX}/{collection}/{movie}"
    parts = [f"{base}/part-{n:04d}.tar" for n in range(len(chunks))]

    def one(index: int) -> None:
        key = parts[index]
        if key in existing:  # uploaded on an earlier run
            with lock:
                progress["skipped_parts"] += 1
            return
        body = pack(movie, chunks[index])
        put_part(s3, key, body)
        with lock:
            progress["sent"] += len(body)
            progress["parts"] += 1

    list(pool.map(one, range(len(chunks))))
    manifest = {"collection": collection, "movie": movie, "frames": len(frames),
                "parts": [p.rsplit("/", 1)[-1] for p in parts]}
    put_part(s3, f"{base}/manifest.json", json.dumps(manifest).encode())
    with lock:
        progress["movies"] += 1
    say(f"stored {collection}/{movie}: {len(frames):,} frames in {len(parts)} parts")


def reporter(total_bytes: int, total_movies: int, stop: threading.Event) -> None:
    last, last_time = 0, time.monotonic()
    while not stop.wait(15):
        now = time.monotonic()
        with lock:
            p = dict(progress)
        rate = (p["sent"] - last) / (now - last_time)
        last, last_time = p["sent"], now
        eta = (total_bytes - p["sent"]) / rate / 3600 if rate > 0 else float("inf")
        say(f"{p['movies']}/{total_movies} movies  {p['parts']} parts  "
            f"{p['sent'] / 1e9:6.2f}/{total_bytes / 1e9:.1f} GB  {rate / 1e6:4.1f} MB/s  ETA {eta:4.1f} h")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", type=int, default=0, help="upload at most this many movies (0 = all)")
    parser.add_argument("--parallel", type=int, default=3, help="parts uploading at once")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    s3 = spaces.client()
    keys = spaces.all_keys(s3)
    stored = spaces.stored_movies(s3, keys)
    local = spaces.local_movies()
    todo = [(c, d) for c, d in local if (c, d.name) not in stored]
    partial = sum(1 for c, d in todo if any(k.startswith(f"{spaces.PREFIX}/{c}/{d.name}/") for k in keys))
    say(f"bucket {spaces.BUCKET} ({spaces.REGION}): {len(stored)}/{len(local)} movies complete, "
        f"{partial} partly uploaded, {sum(keys.values()) / 1e9:.1f} GB stored; {len(todo)} to go")
    if args.status or not todo:
        return
    if args.movies:
        todo = todo[:args.movies]

    total = sum(f.stat().st_size for _, d in todo for f in frames_of(d))
    stop = threading.Event()
    threading.Thread(target=reporter, args=(total, len(todo), stop), daemon=True).start()
    failures = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for collection, directory in todo:  # one movie at a time, its parts in parallel
            try:
                upload_movie(s3, pool, collection, directory, set(keys))
            except Exception as error:  # keep going; a rerun resumes it
                failures.append(directory.name)
                say(f"FAILED {directory.name}: {error}")
    stop.set()
    say(f"finished: {progress['movies']}/{len(todo)} movies, {progress['parts']} parts "
        f"({progress['skipped_parts']} already there), {progress['sent'] / 1e9:.1f} GB in "
        f"{(time.monotonic() - started) / 3600:.2f} h"
        + (f"; unfinished: {', '.join(failures)} (rerun to resume)" if failures else ""))


if __name__ == "__main__":
    main()
