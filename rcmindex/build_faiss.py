"""Build the FAISS search index over rcmindex tiles.

Indexes every frame's 4x4 half-stride tiles (49) and whole-frame embedding
(1): 124M 128-d unit vectors, inner product. IVF with an HNSW coarse
quantizer finds candidate lists quickly; PQ32 stores each vector in 32 bytes
(~4 GB). Vector ids are frame * 64 + view (views 0-48 tiles, 63 whole frame).

Also writes catalog.json: frame names in id order and, per movie, the .npy
files and first frame id, so search can re-score candidates exactly from the
original fp16 vectors (memory-mapped).

    .venv/bin/python -m rcmindex.build_faiss
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from rcmindex.benchmark_crops import index_files

OUT = Path(__file__).resolve().parents[1] / ".local/faiss"
GLOBAL_VIEW = 63
started = time.monotonic()


def note(message: str) -> None:
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def vectors(tiles: np.ndarray, whole: np.ndarray, first: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fp32 vectors and ids for the given rows of one movie (tiles then whole frame)."""
    t = np.asarray(tiles[rows], dtype=np.float32)  # [r, 49, 128]
    r, views, dim = t.shape
    frame = (first + rows).astype(np.int64)
    ids = np.concatenate([(frame[:, None] * 64 + np.arange(views)).ravel(), frame * 64 + GLOBAL_VIEW])
    return np.concatenate([t.reshape(r * views, dim), np.asarray(whole[rows], dtype=np.float32)]), ids


def one_per_film(entries: list) -> list:
    """Some films sit in both collections under slightly different names
    ("Anatomy_Of_A_Murder_1959" / "Anatomy_of_a_Murder_1959"); index one copy,
    or they fill each other's result slots. full-curtain wins (the validation
    suite names it) unless the other copy is a 5%+ longer cut."""
    films: dict[str, list] = {}
    for movie, files, records in entries:
        key = "".join(ch for ch in movie.split("/")[1].lower() if ch.isalnum())
        films.setdefault(key, []).append((movie, files, records, len(np.load(files["4"], mmap_mode="r"))))
    kept = []
    for copies in films.values():
        copies.sort(key=lambda c: (not c[0].startswith("full-curtain/"), c[0]))
        best = copies[0]
        for other in copies[1:]:
            if other[3] >= best[3] * 1.05:
                best = other
        for other in copies:
            if other is not best:
                note(f"duplicate film: keeping {best[0]} ({best[3]:,} frames), dropping {other[0]} ({other[3]:,})")
        kept.append(best)
    return sorted(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", type=int, default=16384)
    parser.add_argument("--pq", type=int, default=32, help="bytes per vector")
    parser.add_argument("--train-frames", type=int, default=30_000)
    args = parser.parse_args()
    faiss.omp_set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)

    movies, names, first = [], [], 0
    for movie, files, records, count in one_per_film(index_files()):
        collection = movie.split("/")[0]
        names += [f"{collection}/{r}" for r in json.loads(records.read_text())]
        movies.append({"movie": movie, "first": first, "frames": count,
                       "tiles": str(files["4"]), "global": str(files["1"])})
        first += count
    total = first * 50
    note(f"{len(movies)} movies, {first:,} frames, {total:,} vectors")
    (OUT / "catalog.json").write_text(json.dumps({"movies": movies, "names": names, "views": 49,
                                                  "global_view": GLOBAL_VIEW, "id_stride": 64}))

    index = faiss.index_factory(128, f"IVF{args.lists}_HNSW32,PQ{args.pq}", faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    sample = []
    for m in movies:
        rows = np.sort(rng.choice(m["frames"], max(1, args.train_frames * m["frames"] // first), replace=False))
        sample.append(vectors(np.load(m["tiles"], mmap_mode="r"), np.load(m["global"], mmap_mode="r"),
                              m["first"], rows)[0])
    sample = np.concatenate(sample)
    note(f"training on {len(sample):,} sampled vectors")
    index.train(sample)
    del sample
    note("trained")

    added = 0
    for number, m in enumerate(movies, 1):
        tiles, whole = np.load(m["tiles"], mmap_mode="r"), np.load(m["global"], mmap_mode="r")
        for start in range(0, m["frames"], 4096):
            x, ids = vectors(tiles, whole, m["first"], np.arange(start, min(start + 4096, m["frames"])))
            index.add_with_ids(x, ids)
            added += len(ids)
        if number % 10 == 0 or number == len(movies):
            note(f"added {number}/{len(movies)} movies, {added:,}/{total:,} vectors")
    path = OUT / f"tiles4-global-ivf{args.lists}-pq{args.pq}.faiss"
    faiss.write_index(index, str(path))
    note(f"wrote {path} ({path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
