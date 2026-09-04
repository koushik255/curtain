from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.model import DEFAULT_MODEL_PATH, PROJECT_ROOT, SSCDModel


DEFAULT_SCREENSHOTS = PROJECT_ROOT / "screenshots"
DEFAULT_INDEX = PROJECT_ROOT / "index"
FRAME_PATTERN = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
MOVIE_YEAR_PATTERN = re.compile(r"^(.*)_(\d{4})$")
# Screenshots were extracted at 2 fps, but the search index intentionally keeps
# every other frame for a 1 fps retrieval index.
SOURCE_FRAME_INTERVAL_SECONDS = 0.5
SAMPLE_INTERVAL_SECONDS = 1.0
FRAME_STRIDE = round(SAMPLE_INTERVAL_SECONDS / SOURCE_FRAME_INTERVAL_SECONDS)


def atomic_json(path: Path, payload: object, *, compact: bool = False) -> None:
    # A PID-specific name lets multiple index workers safely refresh the root
    # manifest at the same time.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if compact:
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
    else:
        temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def movie_details(movie_id: str) -> tuple[str, int | None]:
    match = MOVIE_YEAR_PATTERN.fullmatch(movie_id)
    if match:
        return match.group(1).replace("_", " "), int(match.group(2))
    return movie_id.replace("_", " "), None


def frame_files(movie_dir: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in movie_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.fullmatch(path.name)
        if match:
            frames.append((int(match.group(1)), path.resolve()))
    frames.sort(key=lambda item: item[0])
    return frames


def source_fingerprint(frames: list[tuple[int, Path]]) -> str:
    digest = hashlib.sha256()
    for frame_number, path in frames:
        stat = path.stat()
        digest.update(f"{frame_number}:{path.name}:{stat.st_size}\n".encode())
    return digest.hexdigest()


def stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_records(movie_id: str, frames: list[tuple[int, Path]]) -> list[dict]:
    title, year = movie_details(movie_id)
    return [
        {
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "frame_number": frame_number,
            "timestamp_seconds": round(
                (frame_number - 1) * SOURCE_FRAME_INTERVAL_SECONDS, 3
            ),
            "image_path": stored_path(path),
        }
        for frame_number, path in frames
    ]


def matching_manifest(
    path: Path, fingerprint: str, frame_count: int, storage_dtype: np.dtype
) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        manifest.get("source_fingerprint") == fingerprint
        and manifest.get("frames") == frame_count
        and manifest.get("sample_interval_seconds") == SAMPLE_INTERVAL_SECONDS
        and manifest.get("dimensions") == SSCDModel.dimensions
        and manifest.get("dtype") == storage_dtype.name
    )


def load_images(batch: list[tuple[int, Path]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        for _, path in batch:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
    except Exception:
        for image in images:
            image.close()
        raise
    return images


def index_movie(
    movie_dir: Path,
    frames: list[tuple[int, Path]],
    output_dir: Path,
    model: SSCDModel,
    batch_size: int,
    storage_dtype: np.dtype,
    force: bool,
) -> dict:
    movie_id = movie_dir.name
    title, year = movie_details(movie_id)
    fingerprint = source_fingerprint(frames)
    movie_output = output_dir / "movies" / movie_id
    final_manifest = movie_output / "manifest.json"

    if not force and matching_manifest(
        final_manifest, fingerprint, len(frames), storage_dtype
    ):
        print(f"Skipping {title}: {len(frames):,} frames are already indexed.")
        return json.loads(final_manifest.read_text())

    building = output_dir / ".building" / movie_id
    building.mkdir(parents=True, exist_ok=True)
    partial_embeddings = building / "embeddings.npy"
    state_path = building / "state.json"
    expected_state = {
        "movie_id": movie_id,
        "source_fingerprint": fingerprint,
        "frames": len(frames),
        "dimensions": SSCDModel.dimensions,
        "dtype": storage_dtype.name,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "source_frame_interval_seconds": SOURCE_FRAME_INTERVAL_SECONDS,
    }

    position = 0
    if state_path.is_file() and partial_embeddings.is_file() and not force:
        state = json.loads(state_path.read_text())
        if all(state.get(key) == value for key, value in expected_state.items()):
            position = int(state.get("next_frame", 0))
            embeddings = np.lib.format.open_memmap(partial_embeddings, mode="r+")
            if (
                embeddings.shape != (len(frames), SSCDModel.dimensions)
                or embeddings.dtype != storage_dtype
            ):
                raise RuntimeError(
                    f"The partial index for {movie_id} has the wrong shape or dtype."
                )
            print(f"Resuming {title} at frame {position + 1:,}/{len(frames):,}.")
        else:
            raise RuntimeError(
                f"The screenshots changed while {movie_id} was being indexed. "
                "Run again with --force to restart that movie."
            )
    else:
        embeddings = np.lib.format.open_memmap(
            partial_embeddings,
            mode="w+",
            dtype=storage_dtype,
            shape=(len(frames), SSCDModel.dimensions),
        )
        atomic_json(state_path, {**expected_state, "next_frame": 0})

    started = time.perf_counter()
    resumed_at = position
    while position < len(frames):
        end = min(position + batch_size, len(frames))
        batch = frames[position:end]
        images = load_images(batch)
        try:
            embeddings[position:end] = model.encode(images)
        finally:
            for image in images:
                image.close()
        embeddings.flush()
        position = end
        atomic_json(state_path, {**expected_state, "next_frame": position})

        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = (position - resumed_at) / elapsed
        remaining = (len(frames) - position) / max(rate, 1e-6)
        print(
            f"{title}: {position:,}/{len(frames):,} "
            f"({rate:.1f} frames/s, {remaining / 60:.1f} min remaining)"
        )

    del embeddings
    records = make_records(movie_id, frames)
    records_path = building / "records.json"
    manifest_path = building / "manifest.json"
    atomic_json(records_path, records, compact=True)
    manifest = {
        **expected_state,
        "title": title,
        "year": year,
        "model": model.model_path.name,
        "device": model.device.type,
        "first_frame": frames[0][0],
        "last_frame": frames[-1][0],
    }
    atomic_json(manifest_path, manifest)

    movie_output.mkdir(parents=True, exist_ok=True)
    os.replace(partial_embeddings, movie_output / "embeddings.npy")
    os.replace(records_path, movie_output / "records.json")
    # The manifest moves last, so search never sees a half-finished movie index.
    os.replace(manifest_path, final_manifest)
    state_path.unlink(missing_ok=True)
    building.rmdir()
    print(f"Finished {title}: {len(frames):,} frames.")
    return manifest


def write_root_manifest(output_dir: Path) -> dict:
    movies = []
    for path in sorted((output_dir / "movies").glob("*/manifest.json")):
        try:
            movies.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    manifest = {
        "version": 1,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "movies": len(movies),
        "frames": sum(movie["frames"] for movie in movies),
        "dimensions": SSCDModel.dimensions,
        "movie_indexes": movies,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_index(args: argparse.Namespace) -> None:
    screenshot_root = args.screenshots.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not screenshot_root.is_dir():
        raise RuntimeError(f"Screenshot directory does not exist: {screenshot_root}")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1.")
    if args.limit < 0:
        raise RuntimeError("--limit cannot be negative.")
    storage_dtype = np.dtype(args.dtype)

    selected: list[tuple[Path, list[tuple[int, Path]]]] = []
    remaining = args.limit
    movie_dirs = sorted(
        (path for path in screenshot_root.iterdir() if path.is_dir()),
        reverse=args.reverse,
    )
    for movie_dir in movie_dirs:
        frames = frame_files(movie_dir)
        if not frames:
            continue
        frames = [
            frame
            for frame in frames
            if (frame[0] - 1) % FRAME_STRIDE == 0
        ]
        if args.movie and args.movie.casefold() not in movie_dir.name.casefold():
            continue
        if remaining:
            frames = frames[:remaining]
            remaining -= len(frames)
        selected.append((movie_dir, frames))
        if args.limit and remaining == 0:
            break

    if not selected:
        raise RuntimeError(f"No movie screenshots were found under {screenshot_root}.")

    total = sum(len(frames) for _, frames in selected)
    print(f"Found {len(selected)} movie(s) and {total:,} frame(s).")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SSCDModel(args.model, args.device)
    for movie_dir, frames in selected:
        lock_dir = output_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{movie_dir.name}.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"Skipping {movie_dir.name}: another worker is indexing it.")
                continue
            index_movie(
                movie_dir,
                frames,
                output_dir,
                model,
                args.batch_size,
                storage_dtype,
                args.force,
            )

    manifest = write_root_manifest(output_dir)
    print(
        f"Index ready: {manifest['movies']} movie(s), "
        f"{manifest['frames']:,} frame(s) in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable SSCD movie-frame index.")
    parser.add_argument("--screenshots", type=Path, default=DEFAULT_SCREENSHOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Embedding storage type (float16 uses half the disk space).",
    )
    parser.add_argument("--movie", help="Only index movie folders containing this text.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process movies in reverse order (useful for a second worker).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Index only the first N frames (useful with a separate test output).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected movie indexes, including incompatible partial work.",
    )
    return parser.parse_args()


def main() -> None:
    build_index(parse_args())


if __name__ == "__main__":
    main()
