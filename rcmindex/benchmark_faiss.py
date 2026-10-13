"""Accuracy and latency of rcmindex.search (FAISS + exact re-scoring) on the
same queries as benchmark_crops, so the numbers compare with brute force.

    .venv/bin/python -m rcmindex.benchmark_faiss --queries 400
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from rcmindex import search
from rcmindex.benchmark_crops import NEW_MODEL, SOURCES, crop_like_screenshot, embed, jpeg
from rcmindex.search import pieces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(16)

    t = time.perf_counter()
    engine = search.load()
    names = engine["names"]
    frame_id = {n: i for i, n in enumerate(names)}
    print(f"loaded index + catalog in {time.perf_counter() - t:.0f}s, {engine['index'].ntotal:,} vectors")

    # Same queries as benchmark_crops (same seed and draw order).
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(names)), args.queries)
    crops, fulls, targets = [], [], []
    for i in picks:
        collection, rest = names[i].split("/", 1)
        with Image.open(SOURCES[collection] / rest) as source:
            frame = source.convert("RGB")
        crops.append(crop_like_screenshot(frame, rng))
        fulls.append(jpeg(frame, rng.randint(60, 90)))
        targets.append(i)
    for spec in args.extra:
        path, name = spec.split("=")
        crops.append(Image.open(path).convert("RGB"))
        fulls.append(crops[-1])
        targets.append(frame_id[name])

    model = CurtainEncoder.from_checkpoint(NEW_MODEL)
    crop_q = embed(model, crops).numpy()
    full_q = embed(model, fulls).numpy()
    piece_q = embed(model, [p for c in crops for p in pieces(c)]).numpy().reshape(len(crops), 4, -1)

    def title(i: int) -> str:
        return names[i].split("/")[1]

    def evaluate(label: str, queries: list[np.ndarray], count: int) -> None:
        hits1 = hits5 = near = movie = 0
        times = []
        for q, target in zip(queries[:count], targets[:count]):
            t = time.perf_counter()
            found = search.search(engine, q, candidates=args.candidates)
            times.append(time.perf_counter() - t)
            ids = [f for f, _ in found]
            hits1 += bool(ids) and ids[0] == target
            hits5 += target in ids
            if ids:
                same = title(ids[0]) == title(target)
                movie += same
                near += same and abs(int(names[ids[0]][-10:-4]) - int(names[target][-10:-4])) <= 2
        n = count
        print(f"  {label:<30} exact top-1 {hits1 / n:6.1%}  top-5 {hits5 / n:6.1%}  "
              f"within 2 frames {near / n:6.1%}  right movie {movie / n:6.1%}  "
              f"median {np.median(times) * 1000:5.0f} ms  p95 {np.percentile(times, 95) * 1000:5.0f} ms")

    for nprobe in args.nprobe:
        search.faiss.extract_index_ivf(engine["index"]).nprobe = nprobe
        print(f"\nnprobe {nprobe}, {args.candidates} candidates per query image")
        evaluate("crop", [c[None] for c in crop_q], args.queries)
        evaluate("crop + pieces", [np.concatenate([c[None], p]) for c, p in zip(crop_q, piece_q)], args.queries)
        evaluate("full frame", [f[None] for f in full_q], args.queries)
        for n, spec in enumerate(args.extra):
            col = args.queries + n
            found = search.search(engine, np.concatenate([crop_q[col][None], piece_q[col]]), candidates=args.candidates)
            ids = [f for f, _ in found]
            rank = ids.index(targets[col]) + 1 if targets[col] in ids else ">5"
            print(f"  real screenshot (crop + pieces): top-1 {names[ids[0]]} ({found[0][1]:.3f}), target rank {rank}")


if __name__ == "__main__":
    main()
