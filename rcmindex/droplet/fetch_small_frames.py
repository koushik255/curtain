"""Bring pruned frames back from Spaces as small WebP copies for the site.

prune_frames.py deletes local frames once Spaces holds them; the site still
wants a frame to show and to align against. This streams each stored movie's
tar parts, shrinks every frame (default 640 px wide, WebP quality 75, about a
quarter of the original JPEG) and writes

    .local/frames-small/<collection>/<movie>/<frame>.webp

Movies whose full frames are still local are skipped. Each finished part
leaves a marker, so it can be stopped and rerun any time.

    python3 rcmindex/droplet/fetch_small_frames.py
    python3 rcmindex/droplet/fetch_small_frames.py --width 960 --quality 75
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3  # noqa: F401
    from PIL import Image
except ImportError:  # the system python has neither; rerun through uv
    if os.environ.get("CURTAIN_UV_REEXEC"):
        sys.exit("boto3/pillow unavailable even under uv")
    os.environ["CURTAIN_UV_REEXEC"] = "1"
    os.execvp("uv", ["uv", "run", "--quiet", "--no-project", "--with", "boto3", "--with", "pillow",
                     "python", sys.argv[0], *sys.argv[1:]])

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spaces  # noqa: E402

SMALL = spaces.ROOT / ".local/frames-small"
started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def shrink(data: bytes, width: int, quality: int) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        image.draft("RGB", (width, width))  # JPEG decodes straight at 1/2, 1/4... scale
        image = image.convert("RGB")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=quality, method=2)
    return buffer.getvalue()


def fetch_part(url: str, target: Path, done: Path, width: int, quality: int) -> tuple[int, int, int]:
    """Download one tar part and write its frames small. (frames, bytes in, bytes out)."""
    for attempt in range(1, 6):
        frames = got = wrote = 0
        try:
            # Whole part first (~100 MB, seconds): a stream read slowly while
            # compressing gets dropped by Spaces.
            with urllib.request.urlopen(url, timeout=120) as response:
                body = response.read()
            with tarfile.open(fileobj=io.BytesIO(body), mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".jpg"):
                        continue
                    data = tar.extractfile(member).read()
                    small = shrink(data, width, quality)
                    out = target / (Path(member.name).stem + ".webp")
                    temp = out.with_suffix(".tmp")
                    temp.write_bytes(small)
                    temp.rename(out)
                    frames, got, wrote = frames + 1, got + len(data), wrote + len(small)
            done.touch()
            return frames, got, wrote
        except Exception as error:
            print(f"retry {target.name}/{done.name} ({attempt}/5): {error}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {target.name}/{done.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    s3 = spaces.client()
    keys = spaces.all_keys(s3)
    jobs = []
    for collection, movie in sorted(spaces.stored_movies(s3, keys)):
        base = f"{spaces.PREFIX}/{collection}/{movie}"
        manifest = json.loads(s3.get_object(Bucket=spaces.BUCKET, Key=f"{base}/manifest.json")["Body"].read())
        local = spaces.SOURCES[collection] / movie
        if local.is_dir() and sum(1 for e in os.scandir(local) if e.name.endswith(".jpg")) >= manifest["frames"]:
            continue  # full frames still here
        target = SMALL / collection / movie
        target.mkdir(parents=True, exist_ok=True)
        for part in manifest["parts"]:
            done = target / f".{part}.done"
            if not done.exists():
                url = s3.generate_presigned_url("get_object", Params={"Bucket": spaces.BUCKET, "Key": f"{base}/{part}"},
                                                ExpiresIn=48 * 3600)
                jobs.append((url, target, done, keys[f"{base}/{part}"]))
    total = sum(size for *_, size in jobs)
    say(f"{len(jobs)} parts to fetch ({total / 1e9:.1f} GB) at {args.width} px, WebP q{args.quality}, "
        f"{args.workers} workers")

    frames = got = wrote = parts = 0
    last = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_part, url, target, done, args.width, args.quality)
                   for url, target, done, _ in jobs]
        for future in as_completed(futures):
            try:
                f, g, w = future.result()
            except Exception as error:  # a rerun picks the part up again
                say(f"FAILED: {error}")
                continue
            frames, got, wrote, parts = frames + f, got + g, wrote + w, parts + 1
            if time.monotonic() - last >= 30 or parts == len(jobs):
                last = time.monotonic()
                rate = got / (last - started)
                eta = (total - got) / rate / 3600 if rate else float("inf")
                say(f"{parts}/{len(jobs)} parts  {frames:,} frames  {got / 1e9:6.1f}/{total / 1e9:.1f} GB in "
                    f"-> {wrote / 1e9:5.1f} GB out ({wrote / max(got, 1):.0%})  {rate / 1e6:4.0f} MB/s  ETA {eta:4.1f} h")
    say(f"finished: {parts}/{len(jobs)} parts, {frames:,} frames, {wrote / 1e9:.1f} GB written to {SMALL}")


if __name__ == "__main__":
    main()
