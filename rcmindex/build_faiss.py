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

from rcmindex.benchmark_crops import RUNS, index_files

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


def build_ondisk(trained, movies: list, total: int, out: Path, args) -> None:
    """Add movies to copies of the trained empty index in shards of
    --shard-movies, write each shard, then merge every shard's inverted lists
    into one .ivfdata (OnDiskInvertedLists). Peak RAM is one shard, not the
    whole index. Writes the same <name>.ondisk.faiss + <name>.ivfdata pair as
    ondisk_faiss.py, which search.load() opens with IO_FLAG_READ_ONLY."""
    from faiss.contrib.ondisk import merge_ondisk

    from rcmindex.ondisk_faiss import ondisk_paths

    name = out / f"tiles4-global-ivf{args.lists}-pq{args.pq}.faiss"
    trained_path = out / "trained.faiss"
    faiss.write_index(trained, str(trained_path))
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    shards, added = [], 0
    for s in range(0, len(movies), args.shard_movies):
        path = shard_dir / f"shard-{s // args.shard_movies:03d}.faiss"
        group = movies[s:s + args.shard_movies]
        if not path.exists():  # resumable
            index = faiss.read_index(str(trained_path))
            for m in group:
                tiles, whole = np.load(m["tiles"], mmap_mode="r"), np.load(m["global"], mmap_mode="r")
                for start in range(0, m["frames"], 4096):
                    x, ids = vectors(tiles, whole, m["first"], np.arange(start, min(start + 4096, m["frames"])))
                    index.add_with_ids(x, ids)
            faiss.write_index(index, str(path.with_suffix(".tmp")))
            path.with_suffix(".tmp").rename(path)
            del index
        shards.append(str(path))
        added += sum(m["frames"] for m in group) * 50
        note(f"shard {len(shards)}: {s + len(group)}/{len(movies)} movies, {added:,}/{total:,} vectors")
    index_path, ivfdata = ondisk_paths(name)
    index = faiss.read_index(str(trained_path))
    note(f"merging {len(shards)} shards into {ivfdata}")
    merge_ondisk(index, shards, str(ivfdata))
    faiss.write_index(index, str(index_path))
    note(f"wrote {index_path} and {ivfdata} ({ivfdata.stat().st_size / 1e9:.1f} GB); "
         f"delete {shard_dir} and {trained_path} once it checks out")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", type=int, default=16384)
    parser.add_argument("--pq", type=int, default=32, help="bytes per vector")
    parser.add_argument("--train-frames", type=int, default=30_000)
    # Another model's arrays and index go in their own folders, so the site's
    # .local/faiss and the old arrays stay untouched until it switches.
    parser.add_argument("--runs", type=Path, default=RUNS, help="rcmindex run directories to index")
    parser.add_argument("--out", type=Path, default=OUT, help="where catalog.json and the index go")
    parser.add_argument("--ondisk", action="store_true",
                        help="add in shards of --shard-movies and merge straight into the on-disk index "
                             "search.load() opens (no full in-RAM index; ondisk_faiss.py not needed)")
    parser.add_argument("--shard-movies", type=int, default=40)
    args = parser.parse_args()
    out = args.out
    faiss.omp_set_num_threads(16)
    out.mkdir(parents=True, exist_ok=True)

    movies, names, first = [], [], 0
    for movie, files, records, count in one_per_film(index_files(args.runs)):
        collection = movie.split("/")[0]
        names += [f"{collection}/{r}" for r in json.loads(records.read_text())]
        movies.append({"movie": movie, "first": first, "frames": count,
                       "tiles": str(files["4"]), "global": str(files["1"])})
        first += count
    total = first * 50
    note(f"{len(movies)} movies, {first:,} frames, {total:,} vectors")
    (out / "catalog.json").write_text(json.dumps({"movies": movies, "names": names, "views": 49,
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

    if args.ondisk:
        build_ondisk(index, movies, total, out, args)
        return

    added = 0
    for number, m in enumerate(movies, 1):
        tiles, whole = np.load(m["tiles"], mmap_mode="r"), np.load(m["global"], mmap_mode="r")
        for start in range(0, m["frames"], 4096):
            x, ids = vectors(tiles, whole, m["first"], np.arange(start, min(start + 4096, m["frames"])))
            index.add_with_ids(x, ids)
            added += len(ids)
        if number % 10 == 0 or number == len(movies):
            note(f"added {number}/{len(movies)} movies, {added:,}/{total:,} vectors")
    path = out / f"tiles4-global-ivf{args.lists}-pq{args.pq}.faiss"
    faiss.write_index(index, str(path))
    note(f"wrote {path} ({path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
