"""Does the wider candidate funnel help or hurt? Old vs new search settings.

Runs each query once through FAISS (CANDIDATES tiles per view), then ranks
and aligns it the way the site would under each setting:

  old         128 tiles per view, SIFT on the top 10
  wide        512 tiles, SIFT on the top 10, then up to 40 if none confirmed
  neighbours  wide, plus the 4 frames either side of the 50 best (the site now)
  128+nb      128 tiles with neighbours and the same staged SIFT

A 128-tile search is the first 128 of the 512 (FAISS returns them in order).
SIFT results are shared between settings. Queries: the 110-image validation
suite plus --extra fresh ones per type, as the page uploads them.

    .venv/bin/python -m rcmindex.benchmark_funnel --extra 30
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import align, search
from rcmindex.benchmark_int8 import queries
from rcmindex.validation_suite import LOCAL, SOURCES, browser_upload, embed, score, timestamps

FIRST, MOST = 10, 40  # apps/server.py ALIGN_FIRST, ALIGN_CANDIDATES
SETTINGS = {"old": (128, False, False), "wide": (512, False, True),
            "neighbours": (512, True, True), "128+nb": (128, True, True)}  # tiles, neighbours, staged


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
    times = timestamps()
    items = queries(args.extra, args.seed)
    print(f"{len(items)} queries", flush=True)
    pool = ThreadPoolExecutor(16)

    def path_of(frame: int):
        # As the site does: the original JPEG, else the 640 px WebP copy.
        collection, rest = engine["names"][frame].split("/", 1)
        path = SOURCES[collection] / rest
        if not path.is_file():
            path = (LOCAL / "frames-small" / collection / rest).with_suffix(".webp")
        return path if path.is_file() else None

    rows = []
    started = time.perf_counter()
    for number, (kind, image, target) in enumerate(items, 1):
        image = browser_upload(image)
        q = embed(model, [image, *search.pieces(image)])
        q = np.ascontiguousarray(q / np.linalg.norm(q, axis=1, keepdims=True), dtype=np.float32)
        _, ids = engine["index"].search(q, max(t for t, _, _ in SETTINGS.values()))
        query = align.features(np.asarray(image.convert("L")))
        points: dict[int, int] = {}  # frame id -> agreeing points, shared by the settings

        def check(frames: list[int]) -> None:
            todo = [f for f in frames if f not in points]

            def one(frame: int) -> int:
                path = path_of(frame)
                return align.align(query, align.frame_features(path))[0] if path else 0
            points.update(zip(todo, pool.map(one, todo)))

        row = {"type": kind}
        for name, (tiles, near, staged) in SETTINGS.items():
            sub = ids[:, :tiles]
            frames = np.unique(sub[sub >= 0] // engine["stride"])
            s = search.rescore(engine, q, frames)
            if near:
                best = frames[np.argsort(-s)[:search.NEIGHBOUR_TOP]]
                extra = np.setdiff1d(search.neighbours(engine, best), frames)
                frames, s = np.concatenate([frames, extra]), np.concatenate([s, search.rescore(engine, q, extra)])
            found = frames[np.argsort(-s, kind="stable")][:MOST if staged else FIRST].tolist()
            check(found[:FIRST])
            checked = FIRST
            if staged and not any(points[f] >= align.VERIFIED_POINTS for f in found[:FIRST]):
                check(found)
                checked = len(found)
            order = align.rerank(found[:checked], [(points[f],) for f in found[:checked]])
            top = [engine["names"][found[i]] for i in order[:5]]
            row[name] = {**score(top, target, times), "frames": len(frames), "aligned": checked,
                         "confirmed": bool(order) and points[found[order[0]]] >= align.VERIFIED_POINTS}
        rows.append(row)
        if number % 20 == 0:
            print(f"  {number}/{len(items)} ({time.perf_counter() - started:.0f}s)", flush=True)

    lines = [f"# Candidate funnel: {len(rows)} queries\n", "| metric | " + " | ".join(SETTINGS) + " |",
             "|---|" + "---|" * len(SETTINGS)]
    for metric in ("exact", "moment", "top5", "movie", "confirmed"):
        lines.append(f"| {metric} | " + " | ".join(f"{sum(r[m][metric] for r in rows)}/{len(rows)}" for m in SETTINGS) + " |")
    for metric in ("frames", "aligned"):
        lines.append(f"| mean {metric} | " + " | ".join(f"{np.mean([r[m][metric] for r in rows]):.0f}" for m in SETTINGS) + " |")
    lines += ["", "exact frame by type:", "", "| type | " + " | ".join(SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]
    for kind in dict.fromkeys(r["type"] for r in rows):
        sub = [r for r in rows if r["type"] == kind]
        lines.append(f"| {kind} | " + " | ".join(f"{sum(r[m]['exact'] for r in sub)}/{len(sub)}" for m in SETTINGS) + " |")
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = LOCAL / "benchmarks/results" / f"funnel-{time.strftime('%Y%m%d-%H%M%S')}"
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text(report)
    print(f"saved {out}.json and .md")


if __name__ == "__main__":
    main()
