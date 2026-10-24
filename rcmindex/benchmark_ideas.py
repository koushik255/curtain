"""Search changes that need no retraining, each against the earlier search.

"before" is the search as of the letterbox and sliver readings (512 tiles per
view, neighbours, SIFT on the top 10 then up to 40); each setting adds one
change, and "site" is what apps/server.py runs now:

  refine    once the top frame is confirmed, also align its 4 frames either
            side and keep whichever has the most agreeing points (the shot is
            right but FAISS often ranks a near-identical neighbour first)
  colour    the search.levels reading (auto levels, which also neutralises a
            cast; CURTAIN_COLOUR_READING)
  grid      the 3x3 grid reading (search.grid), for tight crops
  adaptive  if nothing is confirmed, search again wider (nprobe 256, 2048
            tiles per view) and align that top 40 too (not on the site)
  site      colour + grid + refine
  no-colour grid + refine (the site with CURTAIN_COLOUR_READING=0)

Metrics are the funnel benchmark's plus top20 (the right frame anywhere in
the first 20 results, the unaligned ones in search order after the aligned,
as the page shows them).

    CURTAIN_FAISS_DIR=.local/faiss-subset63-new .venv/bin/python -m rcmindex.benchmark_ideas \\
        --checkpoint curtain_models/colour-ft-v1/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import align, search
from rcmindex.benchmark_int8 import queries
from rcmindex.validation_suite import LOCAL, SOURCES, browser_upload, embed, score, timestamps

FIRST, MOST, SHOWN = 10, 40, 20  # apps/server.py ALIGN_FIRST, ALIGN_CANDIDATES, RESULTS
WIDE_NPROBE, WIDE_TILES = 256, 2048
SETTINGS = {  # extra readings, refine, adaptive
    "before": ((), False, False),
    "refine": ((), True, False),
    "colour": (("colour",), False, False),
    "grid": (("grid",), False, False),
    "adaptive": ((), False, True),
    "site": (("colour", "grid"), True, False),
    "no-colour": (("grid",), True, False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", type=int, default=30, help="fresh queries per type on top of the suite")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint", type=Path, default=LOCAL / "trained_models/combined-rocm-50k-v1/best.pt",
                        help="query encoder: the model whose arrays built the index (CURTAIN_FAISS_DIR)")
    args = parser.parse_args()
    torch.set_num_threads(16)
    model = CurtainEncoder.from_checkpoint(args.checkpoint)
    engine = search.load()
    ivf = faiss.extract_index_ivf(engine["index"])
    nprobe = ivf.nprobe
    times = timestamps()
    items = queries(args.extra, args.seed)
    print(f"{len(items)} queries, index {search.FAISS_DIR}", flush=True)
    pool = ThreadPoolExecutor(16)

    def path_of(frame: int):
        collection, rest = engine["names"][frame].split("/", 1)
        path = SOURCES[collection] / rest
        if not path.is_file():
            path = (LOCAL / "frames-small" / collection / rest).with_suffix(".webp")
        return path if path.is_file() else None

    rows = []
    started = time.perf_counter()
    for number, (kind, image, target) in enumerate(items, 1):
        image = browser_upload(image)
        every = search.query_sets(image, colour=True, finer=True)  # earlier readings, then levels, then grid
        flat = embed(model, [view for reading in every for view in reading])
        flat = np.ascontiguousarray(flat / np.linalg.norm(flat, axis=1, keepdims=True), dtype=np.float32)
        bounds = np.cumsum([0, *map(len, every)])
        vectors = [flat[a:b] for a, b in zip(bounds, bounds[1:])]
        base, extra = vectors[:-2], {"colour": vectors[-2], "grid": vectors[-1]}
        query = align.features(np.asarray(image.convert("L")))
        points: dict[int, int] = {}  # frame id -> agreeing points, shared by the settings
        found_cache: dict[tuple, list[int]] = {}

        def check(frames: list[int]) -> None:
            todo = [f for f in frames if f not in points]

            def one(frame: int) -> int:
                path = path_of(frame)
                return align.align(query, align.frame_features(path))[0] if path else 0
            points.update(zip(todo, pool.map(one, todo)))

        def found_for(names: tuple, wide: bool) -> list[int]:
            key = (names, wide)
            if key not in found_cache:
                ivf.nprobe = WIDE_NPROBE if wide else nprobe
                frames, s = search.candidate_frames(engine, base + [extra[n] for n in names],
                                                    WIDE_TILES if wide else search.CANDIDATES)
                ivf.nprobe = nprobe
                found_cache[key] = frames[np.argsort(-s, kind="stable")][:MOST].tolist()
            return found_cache[key]

        def staged(found: list[int]) -> list[int]:
            """Final order as the site shows it: aligned ones reranked, then the rest."""
            check(found[:FIRST])
            checked = FIRST
            if not any(points[f] >= align.VERIFIED_POINTS for f in found[:FIRST]):
                check(found)
                checked = len(found)
            order = align.rerank(found[:checked], [(points[f],) for f in found[:checked]])
            return [found[i] for i in order] + found[checked:]

        row = {"type": kind, "target": target}
        for name, (names, refine, adaptive) in SETTINGS.items():
            found = found_for(names, False)
            ranked = staged(found)
            aligned = sum(f in points for f in found)
            if adaptive and points.get(ranked[0], 0) < align.VERIFIED_POINTS:
                wide = found_for(names, True)
                check(wide)
                pool_ = list(dict.fromkeys(found + wide))
                order = align.rerank(pool_, [(points.get(f, 0),) for f in pool_])
                ranked = [pool_[i] for i in order]
                aligned = len(pool_)
            if refine and ranked and points.get(ranked[0], 0) >= align.VERIFIED_POINTS:
                near = search.neighbours(engine, np.array([ranked[0]])).tolist()
                check(near)
                best = max(near, key=lambda f: (points[f], f == ranked[0]))
                ranked = [best] + [f for f in ranked if f != best]
            top = [engine["names"][f] for f in ranked[:SHOWN]]
            row[name] = {**score(top[:5], target, times), "top20": target in top, "top": top[:3],
                         "aligned": aligned,
                         "confirmed": bool(ranked) and points.get(ranked[0], 0) >= align.VERIFIED_POINTS}
        rows.append(row)
        if number % 20 == 0:
            print(f"  {number}/{len(items)} ({time.perf_counter() - started:.0f}s)", flush=True)

    lines = [f"# No-retrain ideas: {len(rows)} queries, {search.FAISS_DIR.name}\n",
             "| metric | " + " | ".join(SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]
    for metric in ("exact", "moment", "top5", "top20", "movie", "confirmed"):
        lines.append(f"| {metric} | " + " | ".join(f"{sum(r[m][metric] for r in rows)}/{len(rows)}" for m in SETTINGS) + " |")
    lines.append("| mean aligned | " + " | ".join(f"{np.mean([r[m]['aligned'] for r in rows]):.0f}" for m in SETTINGS) + " |")
    lines += ["", "exact frame by type:", "", "| type | " + " | ".join(SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]
    for kind in dict.fromkeys(r["type"] for r in rows):
        sub = [r for r in rows if r["type"] == kind]
        lines.append(f"| {kind} | " + " | ".join(f"{sum(r[m]['exact'] for r in sub)}/{len(sub)}" for m in SETTINGS) + " |")
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = LOCAL / "benchmarks/results" / f"ideas-{time.strftime('%Y%m%d-%H%M%S')}"
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text(report)
    print(f"saved {out}.json and .md")


if __name__ == "__main__":
    main()
