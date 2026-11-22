"""FAISS vs LanceDB as the candidate index, on a subset of movies.

Both get the same vectors (49 tiles + whole frame per frame, ids frame * 64 +
view as in build_faiss.py) and the same compression: IVF with --lists
partitions and 32-byte PQ codes, inner product. The base movies are indexed,
then --add more are appended the way each system allows without a rebuild:

  faiss            add_with_ids on the trained index (no retraining)
  lancedb-new      table.add(); the new rows are searched exhaustively
  lancedb-opt      then table.optimize(), which folds them into the index

Two of the six validation movies are among the added ones, so their queries
only succeed if appended data is searchable. Every system's candidates go
through the site's own scoring (exact re-score from the fp16 arrays) and SIFT
alignment of the top 10. "exact" is brute force over every subset vector:
what a perfect index would return.

Reports build and add times, disk size, recall@CANDIDATES against brute
force, ANN search time per query (5 vectors), and the site's metrics.

    uv run --with lancedb python -m rcmindex.benchmark_lancedb --movies 40 --add 5
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import lancedb
import numpy as np
import pyarrow as pa
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import align, search
from rcmindex.benchmark_int8 import ALIGN_TOP, queries, rank
from rcmindex.build_faiss import vectors
from rcmindex.validation_suite import LOCAL, SOURCES, VALIDATION, browser_upload, embed, score, timestamps
from rcmindex.data_paths import WEBP

OUT = LOCAL / "lancedb-bench"
K = search.CANDIDATES
started = time.monotonic()


def note(message: str) -> None:
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def pick(movies: list[dict], base: int, add: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Base and added movies: 4 validation movies in base, 2 among the added."""
    validation = [m for m in movies if m["movie"] in {f"full-curtain/{v}" for v in VALIDATION}]
    others = [m for m in movies if m not in validation]
    random.Random(seed).shuffle(others)
    return validation[:4] + others[:base - 4], validation[4:] + others[base - 4:base - 4 + add - 2]


def chunks(movie: dict, size: int = 8192):
    tiles, whole = np.load(movie["tiles"], mmap_mode="r"), np.load(movie["global"], mmap_mode="r")
    for start in range(0, movie["frames"], size):
        yield vectors(tiles, whole, movie["first"], np.arange(start, min(start + size, movie["frames"])))


def arrow(x: np.ndarray, ids: np.ndarray) -> pa.RecordBatch:
    vector = pa.FixedSizeListArray.from_arrays(pa.array(x.astype(np.float16).ravel(), type=pa.float16()), 128)
    return pa.RecordBatch.from_arrays([pa.array(ids, type=pa.int64()), vector], ["id", "vector"])


def size(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9 if path.is_dir() else path.stat().st_size / 1e9


def brute_force(movies: list[dict], q: np.ndarray) -> np.ndarray:
    """[len(q), K] ids of the true top-K vectors per query vector."""
    best_s = torch.full((len(q), K), -torch.inf)
    best_i = torch.zeros((len(q), K), dtype=torch.int64)
    qt = torch.from_numpy(q)
    for m in movies:
        for x, ids in chunks(m, 1024):
            top = (qt @ torch.from_numpy(x).T).topk(K, dim=1)  # this chunk's best, then merge
            s = torch.cat([best_s, top.values], dim=1)
            i = torch.cat([best_i, torch.from_numpy(ids)[top.indices]], dim=1)
            merged = s.topk(K, dim=1)
            best_s, best_i = merged.values, torch.gather(i, 1, merged.indices)
    return best_i.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", type=int, default=40, help="base movies indexed first")
    parser.add_argument("--add", type=int, default=5, help="movies appended afterwards")
    parser.add_argument("--lists", type=int, default=4096, help="IVF partitions for both")
    parser.add_argument("--nprobes", default="16,32,64")
    parser.add_argument("--train-frames", type=int, default=30_000)
    parser.add_argument("--extra", type=int, default=30, help="fresh queries per type besides the suite")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    nprobes = [int(n) for n in args.nprobes.split(",")]
    faiss.omp_set_num_threads(16)
    torch.set_num_threads(16)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    engine = search.load()
    base, added = pick(engine["movies"], args.movies, args.add, args.seed)
    every = base + added
    frames = sum(m["frames"] for m in every)
    note(f"{len(base)} base + {len(added)} added movies, {frames:,} frames, {frames * 50:,} vectors")
    note("added: " + ", ".join(m["movie"] for m in added))
    report = {"base": [m["movie"] for m in base], "added": [m["movie"] for m in added],
              "vectors": frames * 50, "lists": args.lists, "times": {}, "sizes": {}}
    times = report["times"]

    # Queries, embedded once.
    model = CurtainEncoder.from_checkpoint(LOCAL / "trained_models/combined-rocm-50k-v1/best.pt")
    items = queries(args.extra, args.seed)
    images, qs = [], []
    for kind, image, target in items:
        image = browser_upload(image)
        q = embed(model, [image, *search.pieces(image)])
        images.append(image)
        qs.append(q / np.linalg.norm(q, axis=1, keepdims=True))
    note(f"{len(items)} queries embedded")
    added_movies = {m["movie"].split("/")[1] for m in added}

    # ---------------------------------------------------------------- FAISS
    rng = np.random.default_rng(0)
    base_frames = sum(m["frames"] for m in base)
    sample = np.concatenate([vectors(np.load(m["tiles"], mmap_mode="r"), np.load(m["global"], mmap_mode="r"),
                                     m["first"], np.sort(rng.choice(m["frames"], max(1, args.train_frames * m["frames"] // base_frames), replace=False)))[0]
                             for m in base])
    t = time.perf_counter()
    index = faiss.index_factory(128, f"IVF{args.lists}_HNSW32,PQ32", faiss.METRIC_INNER_PRODUCT)
    index.train(sample)
    times["faiss_train"] = time.perf_counter() - t
    t = time.perf_counter()
    for m in base:
        for x, ids in chunks(m):
            index.add_with_ids(x, ids)
    times["faiss_add_base"] = time.perf_counter() - t
    t = time.perf_counter()
    for m in added:
        for x, ids in chunks(m):
            index.add_with_ids(x, ids)
    faiss.write_index(index, str(OUT / "faiss.index"))
    times["faiss_append"] = time.perf_counter() - t  # including the write, as a real update would
    report["sizes"]["faiss"] = size(OUT / "faiss.index")
    note(f"faiss: train {times['faiss_train']:.0f}s, add base {times['faiss_add_base']:.0f}s, "
         f"append {len(added)} movies + write {times['faiss_append']:.0f}s")

    # ---------------------------------------------------------------- LanceDB
    db = lancedb.connect(OUT / "lancedb")
    schema = arrow(np.zeros((1, 128), np.float32), np.zeros(1, np.int64)).schema
    t = time.perf_counter()
    table = db.create_table("tiles", data=(arrow(x, ids) for m in base for x, ids in chunks(m)), schema=schema)
    times["lancedb_write_base"] = time.perf_counter() - t
    t = time.perf_counter()
    table.create_index(metric="dot", num_partitions=args.lists, num_sub_vectors=32, num_bits=8,
                       vector_column_name="vector", index_type="IVF_PQ")
    times["lancedb_index"] = time.perf_counter() - t
    t = time.perf_counter()
    for m in added:
        table.add(pa.Table.from_batches([arrow(x, ids) for x, ids in chunks(m)]))
    times["lancedb_append"] = time.perf_counter() - t
    note(f"lancedb: write {times['lancedb_write_base']:.0f}s, index {times['lancedb_index']:.0f}s, "
         f"append {times['lancedb_append']:.0f}s")

    # ---------------------------------------------------------------- search
    def faiss_search(nprobe: int):
        def run(q: np.ndarray) -> np.ndarray:
            faiss.extract_index_ivf(index).nprobe = nprobe  # set per call: the systems share one index
            return index.search(q, K)[1]
        return run

    def lance_search(nprobe: int):
        def run(q: np.ndarray) -> np.ndarray:
            r = (table.search(q, vector_column_name="vector").distance_type("dot").nprobes(nprobe)
                 .limit(K).select(["id"]).to_arrow())
            out = np.full((len(q), K), -1, dtype=np.int64)
            qi, ids = r["query_index"].to_numpy(), r["id"].to_numpy()
            for i in range(len(q)):
                got = ids[qi == i]
                out[i, :len(got)] = got
            return out
        return run

    note("brute force ground truth")
    t = time.perf_counter()
    truth = brute_force(every, np.concatenate(qs).astype(np.float32))
    note(f"brute force done in {time.perf_counter() - t:.0f}s")
    offsets = np.cumsum([0] + [len(q) for q in qs])

    stamps = timestamps()
    pool = ThreadPoolExecutor(8)
    checked: list[dict[int, int]] = [{} for _ in items]  # per query: frame id -> agreeing points

    def points(n: int, frame: int) -> int:
        collection, rest = engine["names"][frame].split("/", 1)
        path = SOURCES[collection] / rest
        if not path.is_file():  # original pruned: the site aligns against the 640 px WebP copy
            path = (WEBP / collection / rest).with_suffix(".webp")
        query = align.features(np.asarray(images[n].convert("L")))
        return align.align(query, align.frame_features(path))[0]

    def evaluate(label: str, find) -> dict:
        rows, latency, recall = [], [], []
        for n, ((kind, _image, target), q) in enumerate(zip(items, qs)):
            t = time.perf_counter()
            ids = find(q.astype(np.float32)) if find != "exact" else truth[offsets[n]:offsets[n + 1]]
            latency.append(time.perf_counter() - t)
            true = truth[offsets[n]:offsets[n + 1]]
            recall.append(np.mean([len(np.intersect1d(a, b)) / K for a, b in zip(ids, true)]))
            frames_ = np.unique(ids[ids >= 0] // engine["stride"])
            order, _ = rank(engine, search.frame_vectors(engine, frames_), frames_, q)
            todo = [f for f in order[:ALIGN_TOP].tolist() if f not in checked[n]]
            checked[n].update(zip(todo, pool.map(lambda f: points(n, f), todo)))
            top = order[:ALIGN_TOP]
            order = top[align.rerank(top, [(checked[n][f],) for f in top.tolist()])]
            found = [engine["names"][f] for f in order[:5]]
            rows.append({"type": kind, "added": target.split("/")[1] in added_movies,
                         **score(found, target, stamps)})
        result = {"recall": float(np.mean(recall)), "median_ms": float(np.median(latency) * 1000),
                  "p95_ms": float(np.percentile(latency, 95) * 1000), "rows": rows}
        note(f"{label}: recall@{K} {result['recall']:.3f}, {result['median_ms']:.1f} ms, "
             f"exact {sum(r['exact'] for r in rows)}/{len(rows)}")
        return result

    systems = {"exact (brute force)": "exact"}
    for nprobe in nprobes:
        systems[f"faiss nprobe {nprobe}"] = faiss_search(nprobe)
    # Unoptimized, the appended rows are scanned in full (seconds per query): one setting is enough.
    middle = nprobes[len(nprobes) // 2]
    systems[f"lancedb-new nprobe {middle}"] = lance_search(middle)
    results = {label: evaluate(label, find) for label, find in systems.items()}

    t = time.perf_counter()
    table.optimize()
    times["lancedb_optimize"] = time.perf_counter() - t
    report["sizes"]["lancedb"] = size(OUT / "lancedb")
    note(f"lancedb optimize {times['lancedb_optimize']:.0f}s")
    for nprobe in nprobes:
        label = f"lancedb-opt nprobe {nprobe}"
        results[label] = evaluate(label, lance_search(nprobe))

    # ---------------------------------------------------------------- report
    report["results"] = results
    lines = [f"# FAISS vs LanceDB: {len(base)} base + {len(added)} added movies, {frames * 50:,} vectors, "
             f"IVF{args.lists} + PQ32, {len(items)} queries", "",
             "| | FAISS | LanceDB |", "|---|---|---|",
             f"| build (train + index base) | {times['faiss_train'] + times['faiss_add_base']:.0f} s | "
             f"{times['lancedb_write_base'] + times['lancedb_index']:.0f} s (write {times['lancedb_write_base']:.0f} + index {times['lancedb_index']:.0f}) |",
             f"| append {len(added)} movies | {times['faiss_append']:.0f} s (incl. write) | "
             f"{times['lancedb_append']:.0f} s + optimize {times['lancedb_optimize']:.0f} s |",
             f"| on disk | {report['sizes']['faiss']:.1f} GB | {report['sizes']['lancedb']:.1f} GB (incl. fp16 vectors) |",
             "", f"| system | recall@{K} | median ms | p95 ms | exact | moment | movie | exact, added movies |",
             "|---|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        rows = r["rows"]
        new = [x for x in rows if x["added"]]
        lines.append(f"| {label} | {r['recall']:.3f} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | "
                     f"{sum(x['exact'] for x in rows)}/{len(rows)} | {sum(x['moment'] for x in rows)} | "
                     f"{sum(x['movie'] for x in rows)} | {sum(x['exact'] for x in new)}/{len(new)} |")
    text = "\n".join(lines)
    print("\n" + text)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = LOCAL / "benchmarks/results"
    (results_dir / f"lancedb-{stamp}.json").write_text(json.dumps(report, indent=1))
    (results_dir / f"lancedb-{stamp}.md").write_text(text + "\n")
    note(f"saved {results_dir}/lancedb-{stamp}.md")


if __name__ == "__main__":
    main()
