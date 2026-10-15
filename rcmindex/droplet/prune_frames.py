"""Free local disk: delete frames of movies that are safely stored in Spaces.

A movie's frames are deleted only after its Spaces copy checks out: the
manifest lists the same number of frames as the local folder, every tar part
is exactly the size these local files pack to, and one random frame read back
from its part is byte-identical. The site's example images stay local:

    - the validation movies (whole folders; the benchmarks re-run on them)
    - every frame shown on the benchmarks page
    - the gallery frames (2 per movie, 16 when one movie is picked), both for
      the current FAISS catalog and for each folder on its own

    python3 rcmindex/droplet/prune_frames.py            # check only, report what would go
    python3 rcmindex/droplet/prune_frames.py --delete
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spaces  # noqa: E402
from upload_frames import PART_FRAMES, frames_of  # noqa: E402

spaces.ensure_boto3()

LOCAL = spaces.ROOT / ".local"
BENCHMARKS = LOCAL / "benchmarks"
CATALOG = LOCAL / "faiss/catalog.json"
GALLERY_COUNTS = (2, 16)  # apps/server.py gallery: all movies, one movie
EXTRA = {"full-curtain/All_The_Presidents_Men_1976/frame_000977.jpg"}  # benchmark_crops example

started = time.monotonic()
lock = threading.Lock()


def say(message: str) -> None:
    print(f"[{time.monotonic() - started:5.0f}s] {message}", flush=True)


def gallery_positions(frames: int) -> list[int]:
    # np.linspace(0, frames - 1, count + 2, dtype=int)[1:-1], as the gallery does.
    positions = set()
    for count in GALLERY_COUNTS:
        count = min(count, frames)
        step = (frames - 1) / (count + 1)
        positions.update(int(i * step) for i in range(1, count + 1))
    return sorted(positions)


def keepers() -> tuple[set[str], set[str]]:
    """(whole movies "<collection>/<movie>", single frames "<collection>/<movie>/<frame>.jpg")."""
    manifest = json.loads((BENCHMARKS / "validation-v1/manifest.json").read_text())
    whole = {f"full-curtain/{movie}" for movie in manifest["movies"]}
    frames = set(EXTRA) | {q["target"] for q in manifest["queries"]}
    for run in BENCHMARKS.glob("results/*.json"):
        for system in json.loads(run.read_text())["systems"].values():
            for row in system["rows"]:
                frames.update(row.get("found", []))

    # Gallery over the live catalog: films in both folders share one list of spans.
    catalog = json.loads(CATALOG.read_text())
    names, films = catalog["names"], {}
    for entry in catalog["movies"]:
        films.setdefault(entry["movie"].split("/")[1].casefold(), []).append((entry["first"], entry["frames"]))
    for spans in films.values():
        for position in gallery_positions(sum(count for _, count in spans)):
            for start, count in spans:
                if position < count:
                    frames.add(names[start + position])
                    break
                position -= count

    # Gallery over each folder alone, for movies the next catalog adds.
    for collection, directory in spaces.local_movies():
        listing = frames_of(directory)
        frames.update(f"{collection}/{directory.name}/{listing[p].name}" for p in gallery_positions(len(listing)))
    return whole, frames


def part_layout(movie: str, chunk: list[Path]) -> tuple[int, list[tuple[int, int]]]:
    """Size of the tar upload_frames.pack() makes from these files, and each member's (offset, size)."""
    tar = tarfile.open(fileobj=io.BytesIO(), mode="w", format=tarfile.GNU_FORMAT)
    offset, members = 0, []
    for frame in chunk:
        info = tar.gettarinfo(frame, arcname=f"{movie}/{frame.name}")
        offset += len(info.tobuf(tarfile.GNU_FORMAT, tar.encoding, tar.errors))
        members.append((offset, info.size))
        offset += -(-info.size // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
    offset += 2 * tarfile.BLOCKSIZE  # end-of-archive blocks
    return -(-offset // tarfile.RECORDSIZE) * tarfile.RECORDSIZE, members


def verify(s3, keys: dict[str, int], collection: str, directory: Path) -> str | None:
    """None when the Spaces copy of this movie matches the local frames, else why not."""
    movie = directory.name
    base = f"{spaces.PREFIX}/{collection}/{movie}"
    manifest = json.loads(s3.get_object(Bucket=spaces.BUCKET, Key=f"{base}/manifest.json")["Body"].read())
    frames = frames_of(directory)
    if manifest["frames"] != len(frames):
        return f"manifest has {manifest['frames']} frames, folder has {len(frames)}"
    chunks = [frames[i:i + PART_FRAMES] for i in range(0, len(frames), PART_FRAMES)]
    if len(manifest["parts"]) != len(chunks):
        return f"manifest has {len(manifest['parts'])} parts, expected {len(chunks)}"
    layouts = []
    for part, chunk in zip(manifest["parts"], chunks):
        size, members = part_layout(movie, chunk)
        stored = keys.get(f"{base}/{part}")
        if stored != size:
            return f"{part} is {stored} bytes in Spaces, local frames pack to {size}"
        layouts.append(members)
    n = random.randrange(len(frames))  # one frame read back from the bucket
    offset, size = layouts[n // PART_FRAMES][n % PART_FRAMES]
    body = s3.get_object(Bucket=spaces.BUCKET, Key=f"{base}/{manifest['parts'][n // PART_FRAMES]}",
                         Range=f"bytes={offset}-{offset + size - 1}")["Body"].read()
    if body != frames[n].read_bytes():
        return f"{frames[n].name} differs from its copy in Spaces"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="delete (default: only check and report)")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    s3 = spaces.client()
    keys = spaces.all_keys(s3)
    stored = spaces.stored_movies(s3, keys)
    whole, keep = keepers()
    candidates = [(c, d) for c, d in spaces.local_movies()
                  if (c, d.name) in stored and f"{c}/{d.name}" not in whole]
    say(f"{len(candidates)} local movies are stored in Spaces; keeping {len(whole)} validation movies "
        f"and {len(keep):,} example frames")

    totals = {"movies": 0, "files": 0, "bytes": 0}
    problems = []

    def one(item: tuple[str, Path]) -> None:
        collection, directory = item
        try:
            problem = verify(s3, keys, collection, directory)
        except Exception as error:
            problem = f"check failed: {error}"
        if problem:
            with lock:
                problems.append(f"{collection}/{directory.name}: {problem}")
            say(f"KEEP {directory.name}: {problem}")
            return
        doomed = [f for f in frames_of(directory) if f"{collection}/{directory.name}/{f.name}" not in keep]
        size = sum(f.stat().st_size for f in doomed)
        if args.delete:
            for frame in doomed:
                frame.unlink()
        with lock:
            totals["movies"] += 1
            totals["files"] += len(doomed)
            totals["bytes"] += size
            done = totals["movies"]
        if done % 25 == 0:
            say(f"{done}/{len(candidates)} movies verified, {totals['bytes'] / 1e9:.1f} GB "
                + ("freed" if args.delete else "can go"))

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, candidates))
    say(f"{totals['movies']} movies verified: {totals['files']:,} frames, {totals['bytes'] / 1e9:.1f} GB "
        + ("deleted" if args.delete else "would be deleted (rerun with --delete)")
        + (f"; {len(problems)} kept because their Spaces copy did not check out" if problems else ""))


if __name__ == "__main__":
    main()
