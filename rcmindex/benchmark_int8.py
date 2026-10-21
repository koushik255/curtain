"""Does storing the re-score vectors as int8 instead of fp16 change search results?

Search re-scores FAISS's candidate frames from their 50 full vectors
(rcmindex/search.py). This runs that search once per query, then scores the
same candidates from:

  fp16        the stored vectors, as the site does today
  int8        every component times 254, rounded and clipped to [-127, 127]
              (one fixed scale: components of these unit vectors stay under 0.46)
  int8-row    the same with a scale per vector (127 / its largest component),
              which needs a 2-byte scale beside each 128-byte vector

Rounding the vectors and dividing back gives exactly the scores int8 storage
would, so nothing is converted on disk. Reports ranking metrics (no SIFT
alignment, or with --align the site's SIFT re-ranking of the top 10) per
mode, how often the top 10 (the frames alignment checks)
differ from fp16's, and how far scores move.

Queries: the 110-image validation suite, plus --extra fresh queries per
type made the same way from the validation movies.

    .venv/bin/python -m rcmindex.benchmark_int8 --extra 30 --align
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from rcmindex import align, search
from rcmindex.validation_suite import (LOCAL, QUERY_TYPES, SOURCES, SUITE, VALIDATION, browser_upload,
                                       embed, score, timestamps)

ALIGN_TOP = 10


def int8(vectors: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vectors * 254), -127, 127) / 254


def int8_row(vectors: np.ndarray) -> np.ndarray:
    scale = 127 / np.abs(vectors).max(axis=-1, keepdims=True)
    return np.round(vectors * scale) / scale


MODES = {"fp16": lambda v: v, "int8": int8, "int8-row": int8_row}


def queries(extra: int, seed: int) -> list[tuple[str, Image.Image, str]]:
    manifest = json.loads((SUITE / "manifest.json").read_text())["queries"]
    out = [(q["type"], Image.open(SUITE / q["image"]).convert("RGB"), q["target"]) for q in manifest]
    rng = random.Random(seed)
    frames = {m: sorted((SOURCES["full-curtain"] / m).glob("*.jpg")) for m in VALIDATION}
    for kind, build in QUERY_TYPES.items():
        for n in range(extra):
            movie = VALIDATION[n % len(VALIDATION)]
            source = rng.choice(frames[movie])
            with Image.open(source) as image:
                out.append((kind, build(image.convert("RGB"), rng), f"full-curtain/{movie}/{source.name}"))
    return out


def rank(engine: dict, vectors: np.ndarray, frames: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The scoring rule of search.search, on given candidate vectors.
    best = (vectors @ q.T).max(axis=1)
    score_ = np.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(axis=1))
    order = np.argsort(-score_, kind="stable")
    return frames[order], score_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", type=int, default=30, help="fresh queries per type on top of the suite")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--align", action="store_true", help="re-rank each mode's top 10 by SIFT, as the site does")
    args = parser.parse_args()
    torch.set_num_threads(16)
    model = CurtainEncoder.from_checkpoint(LOCAL / "trained_models/combined-rocm-50k-v1/best.pt")
    engine = search.load()
    times = timestamps()
    items = queries(args.extra, args.seed)
    print(f"{len(items)} queries", flush=True)

    pool = ThreadPoolExecutor(8)
    rows = []
    started = time.perf_counter()
    for number, (kind, image, target) in enumerate(items, 1):
        image = browser_upload(image)
        q = embed(model, [image, *search.pieces(image)])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        _, ids = engine["index"].search(np.ascontiguousarray(q, dtype=np.float32), search.CANDIDATES)
        frames = np.unique(ids[ids >= 0] // engine["stride"])
        vectors = search.frame_vectors(engine, frames)
        row = {"type": kind}
        if args.align:
            query = align.features(np.asarray(image.convert("L")))
            checked: dict[int, int] = {}  # frame id -> agreeing points, shared by the modes

            def points(frame: int) -> int:
                # As the site does: the original JPEG, else the 640 px WebP copy of a pruned frame.
                collection, rest = engine["names"][frame].split("/", 1)
                path = SOURCES[collection] / rest
                if not path.is_file():
                    path = (LOCAL / "frames-small" / collection / rest).with_suffix(".webp")
                if not path.is_file():
                    return 0
                return align.align(query, align.frame_features(path))[0]
        base_order = base_score = None  # fp16's final order and scores (MODES starts with fp16)
        for mode, quantize in MODES.items():
            order, s = rank(engine, quantize(vectors), frames, q)
            if args.align:
                todo = [f for f in order[:ALIGN_TOP].tolist() if f not in checked]
                checked.update(zip(todo, pool.map(points, todo)))
                top = order[:ALIGN_TOP]
                order = np.concatenate([top[align.rerank(top, [(checked[f],) for f in top.tolist()])],
                                        order[ALIGN_TOP:]])
            if base_order is None:
                base_order, base_score = order, s
            found = [engine["names"][f] for f in order[:5]]
            row[mode] = {**score(found, target, times),
                         "same_top1": bool(order[0] == base_order[0]),
                         "same_top10": bool(np.array_equal(order[:ALIGN_TOP], base_order[:ALIGN_TOP])),
                         "same_top10_set": set(order[:ALIGN_TOP].tolist()) == set(base_order[:ALIGN_TOP].tolist()),
                         "max_delta": float(np.abs(s - base_score).max())}
        rows.append(row)
        if number % 50 == 0:
            print(f"  {number}/{len(items)} ({time.perf_counter() - started:.0f}s)", flush=True)

    types = list(dict.fromkeys(r["type"] for r in rows))
    print(f"\n{len(rows)} queries ({len(rows) - len(types) * args.extra} suite + {args.extra}/type fresh)\n")
    print("| metric | " + " | ".join(MODES) + " |\n|---|" + "---|" * len(MODES))
    for metric in ("exact", "moment", "top5", "movie", "same_top1", "same_top10", "same_top10_set"):
        print(f"| {metric} | " + " | ".join(f"{sum(r[m][metric] for r in rows)}/{len(rows)}" for m in MODES) + " |")
    for m in MODES:
        deltas = np.array([r[m]["max_delta"] for r in rows])
        print(f"{m}: score change per query, median of max {np.median(deltas):.5f}, worst {deltas.max():.5f}")
    print("\nexact frame by type:")
    for kind in types:
        sub = [r for r in rows if r["type"] == kind]
        print(f"  {kind:12s} " + "  ".join(f"{m} {sum(r[m]['exact'] for r in sub)}/{len(sub)}" for m in MODES))
    out = LOCAL / "benchmarks/results" / f"int8-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
