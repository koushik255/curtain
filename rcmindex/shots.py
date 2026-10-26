"""Shot maps: where each film's camera cuts are, for searching by shot.

A cut is one PySceneDetect's AdaptiveDetector finds in the film's 2 fps frames
that the index's whole-frame vectors confirm (consecutive similarity below
CUT within a frame). On The Godfather Part II either alone mostly cut on
people walking past the camera (the vectors) or on lighting changes and fades
(PySceneDetect); together they gave 1,394 shots, median 5.5 s.

Each film's map is .local/shots/<collection>/<movie>.json: {"cuts": [frame
rows where a new shot starts], ...}. Build them for every film in an index:

    .venv/bin/python -m rcmindex.shots --index .local/faiss-all-films-v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / ".local/shots"
FRAME_DIRS = {"full-curtain": ROOT / ".local/screenshots", "l4-50k-lbfive": Path.home() / "lbfive/screenshots"}
SMALL_FRAMES = ROOT / ".local/frames-small"
CUT = 0.6


def frame_pattern(movie: str) -> str | None:
    """OpenCV image-sequence pattern for "<collection>/<movie>": the originals, else the WebP copies."""
    collection, name = movie.split("/")
    for folder, suffix in ((FRAME_DIRS[collection] / name, "jpg"), (SMALL_FRAMES / movie, "webp")):
        if (folder / f"frame_000001.{suffix}").exists():
            return str(folder / f"frame_%06d.{suffix}")
    return None


def detect(pattern: str, whole: np.ndarray, cut: float = CUT) -> tuple[list[int], int]:
    """(confirmed cuts, cuts PySceneDetect found) for one film's frames."""
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(adaptive_threshold=1.5, min_scene_len=1, window_width=1,
                                          min_content_val=10))
    manager.detect_scenes(open_video(pattern, framerate=2.0, backend="opencv"))
    detected = [start.get_frames() for start, _ in manager.get_scene_list()[1:]]
    whole = whole.astype(np.float32)
    whole /= np.linalg.norm(whole, axis=1, keepdims=True)
    drop = set((np.flatnonzero((whole[1:] * whole[:-1]).sum(axis=1) < cut) + 1).tolist())
    return [c for c in detected if drop & {c - 1, c, c + 1}], len(detected)


def path(movie: str) -> Path:
    return SHOTS / f"{movie}.json"


def load(movie: str, frames: int) -> np.ndarray | None:
    """Shot number of each of the film's frames, or None without a map."""
    p = path(movie)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data["frames"] != frames:
        return None
    start = np.zeros(frames, dtype=np.int32)
    start[data["cuts"]] = 1
    return np.cumsum(start, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True, help="a build_faiss output folder (its catalog.json)")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    movies = json.loads((args.index / "catalog.json").read_text())["movies"]
    for number, m in enumerate(movies, 1):
        out = path(m["movie"])
        if out.exists() and not args.redo:
            continue
        pattern = frame_pattern(m["movie"])
        if pattern is None:
            print(f"{number}/{len(movies)} {m['movie']}: no frames found, skipped", flush=True)
            continue
        started = time.perf_counter()
        whole = np.load(ROOT / m["global"])
        if len(whole) != m["frames"]:
            raise RuntimeError(f"{m['movie']}: {len(whole)} vectors for {m['frames']} frames")
        cuts, detected = detect(pattern, whole)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fps": 2, "frames": m["frames"], "cuts": cuts, "detected": detected,
                                   "cut": CUT, "vectors": m["global"]}))
        print(f"{number}/{len(movies)} {m['movie']}: {len(cuts) + 1} shots "
              f"({detected} detected) in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
