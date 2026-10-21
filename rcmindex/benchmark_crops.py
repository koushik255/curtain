"""Compare the live site's search with rcmindex tile indexes on crop queries.

Live:  crop-v1 model query vs one whole-frame embedding per frame (2.48M).
New:   combined model query vs per-frame tiles from rcmindex: 4x4 grid
       (49 tiles), optional 3x3 (25) and 2x2 (9) grids, and the whole frame.
       A frame scores its best tile. "+ pieces" also searches four
       overlapping corner pieces of the query and averages the five scores.

Queries are random frames turned into screenshot-like crops (25-50% of the
width, 40-80% of the height, random position) and full frames, re-encoded
as JPEG. Exact brute force over every frame.

    .venv/bin/python -m rcmindex.benchmark_crops --queries 400 \
        --extra elevator.png=full-curtain/All_The_Presidents_Men_1976/frame_000977.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import reference_transform
from rcmindex.search import pieces

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
LIVE_MODEL = LOCAL / "trained_models/l4-50k-crop-v1-20260909/best.pt"
LIVE_INDEXES = {"full-curtain": LOCAL / "trained_indexes/l4-50k-crop-v1",
                "l4-50k-lbfive": LOCAL / "trained_indexes/l4-50k-lbfive-crop-v1"}
NEW_MODEL = LOCAL / "trained_models/combined-rocm-50k-v1/best.pt"
RUNS = LOCAL / "rcmindex-runs"
SOURCES = {"full-curtain": LOCAL / "screenshots", "l4-50k-lbfive": Path.home() / "lbfive/screenshots"}
GRIDS = {"4": ".npy", "3": ".g3.npy", "2": ".g2.npy", "1": ".global.npy"}
TOP = 5
PIECES = 4


def index_files(runs: Path = RUNS) -> list[tuple[str, dict[str, Path], Path]]:
    """Per movie: ("<collection>/<movie>", {grid: newest .npy}, records path).
    Some titles exist in both collections, so the collection is part of the key."""
    found: dict[str, dict[str, Path]] = {}
    records: dict[str, Path] = {}
    for path in sorted(runs.glob("*/*/*.npy")):  # run dirs sort by time, newest last
        # Longest suffix first: titles can contain dots ("Mr._Smith...").
        grid = next((g for g, suffix in sorted(GRIDS.items(), key=lambda kv: -len(kv[1]))
                     if path.name.endswith(suffix)), None)
        if grid:
            name = path.name[: -len(GRIDS[grid])]
            movie = f"{path.parent.name}/{name}"
            found.setdefault(movie, {})[grid] = path
            records[movie] = path.parent / f"{name}.records.json"
    return [(m, found[m], records[m]) for m in sorted(found)]


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def crop_like_screenshot(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    cw, ch = int(w * rng.uniform(0.25, 0.5)), int(h * rng.uniform(0.4, 0.8))
    left, top = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return jpeg(image.crop((left, top, left + cw, top + ch)), rng.randint(60, 90))


def embed(model: CurtainEncoder, images: list[Image.Image]) -> torch.Tensor:
    transform = reference_transform(model.image_size)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), 256):
            out.append(model(torch.stack([transform(i) for i in images[start:start + 256]])))
    return torch.cat(out).float()


class TopK:
    """Running top-K (score, frame id) per query column."""

    def __init__(self, columns: int):
        self.scores = torch.full((columns, TOP), -torch.inf)
        self.ids = torch.full((columns, TOP), -1, dtype=torch.long)

    def add(self, scores: torch.Tensor, ids: torch.Tensor) -> None:  # scores [frames, columns]
        k = min(TOP, len(scores))
        s, i = scores.topk(k, dim=0)
        merged_s = torch.cat([self.scores, s.T], 1)
        merged_i = torch.cat([self.ids, ids[i].T], 1)
        order = merged_s.argsort(1, descending=True)[:, :TOP]
        self.scores, self.ids = merged_s.gather(1, order), merged_i.gather(1, order)


def best_view(tiles: np.ndarray, queries: torch.Tensor) -> torch.Tensor:
    """[n, views, 128] or [n, 128] fp16 -> best view score per frame, [n, columns]."""
    block = torch.from_numpy(np.array(tiles, dtype=np.float32))
    if block.dim() == 2:
        return block @ queries.T
    n, views, dim = block.shape
    return (block.reshape(n * views, dim) @ queries.T).reshape(n, views, -1).amax(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--extra", type=str, nargs="*", default=[],
                        help="real screenshots as path=movie/frame_000977.jpg")
    args = parser.parse_args()
    torch.set_num_threads(16)
    rng = random.Random(args.seed)
    started = time.perf_counter()

    def note(message: str) -> None:
        print(f"[{time.perf_counter() - started:5.0f}s] {message}", flush=True)

    movies = index_files()
    names: list[str] = []
    offsets = []
    for movie, _, records in movies:  # names: "<collection>/<movie>/<frame>.jpg"
        offsets.append(len(names))
        collection = movie.split("/")[0]
        names += [f"{collection}/{r}" for r in json.loads(records.read_text())]
    frame_id = {n: i for i, n in enumerate(names)}
    grids = sorted(set.intersection(*(set(files) for _, files, _ in movies)), reverse=True)
    note(f"{len(movies)} movies, {len(names):,} frames, grids in every movie: {', '.join(grids)}")

    # Queries: crop + its pieces, and the full frame.
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
        path, name = spec.split("=")  # name: <collection>/<movie>/<frame>.jpg
        crops.append(Image.open(path).convert("RGB"))
        fulls.append(crops[-1])
        targets.append(frame_id[name])
    targets = torch.tensor(targets)
    q = len(targets)
    piece_images = [p for crop in crops for p in pieces(crop)]  # [q * PIECES], grouped per query

    live_q = embed(CurtainEncoder.from_checkpoint(LIVE_MODEL), crops + fulls)
    new_q = embed(CurtainEncoder.from_checkpoint(NEW_MODEL), crops + fulls + piece_images)
    note(f"embedded {len(crops) + len(fulls) + len(piece_images)} query images")
    crop_cols, piece_cols = slice(0, q), slice(2 * q, None)

    # Live index, mapped onto the shared numbering.
    live = TopK(2 * q)
    for collection, root in LIVE_INDEXES.items():
        for movie_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            records = json.loads((movie_dir / "records.json").read_text())
            ids = torch.tensor([frame_id.get(f"{collection}/{r['movie']}/{r['filename']}", -1) for r in records])
            scores = torch.from_numpy(np.load(movie_dir / "embeddings.npy")).float() @ live_q.T
            scores[ids < 0] = -torch.inf  # frames not in the new index
            live.add(scores, ids)
    note("searched live index")

    # New index. Methods: which grids count, and whether pieces are used.
    methods = {"4x4 + whole": ["4", "1"], "all grids": grids}
    found = {name: TopK(2 * q) for name in methods}
    found.update({f"{name} + pieces": TopK(q) for name in methods})
    block = 1024
    for number, ((movie, files, _), first) in enumerate(zip(movies, offsets), 1):
        arrays = {g: np.load(files[g], mmap_mode="r") for g in grids}
        n = len(arrays["4"])
        for start in range(0, n, block):
            best = {g: best_view(a[start:start + block], new_q) for g, a in arrays.items()}
            ids = torch.arange(start, min(start + block, n)) + first
            for name, use in methods.items():
                top = torch.stack([best[g] for g in use if g in best]).amax(0)  # [frames, all columns]
                found[name].add(top[:, :2 * q], ids)
                piece_scores = top[:, piece_cols].reshape(len(ids), q, PIECES)
                combined = (top[:, crop_cols] + piece_scores.sum(2)) / (1 + PIECES)
                found[f"{name} + pieces"].add(combined, ids)
        if number % 20 == 0:
            note(f"tiles: {first + n:,}/{len(names):,} frames")
    note("searched new index\n")

    def row(label: str, top: TopK, cols: slice, kind: str) -> None:
        ids, t = top.ids[cols], targets[: cols.stop - cols.start]
        if cols.start >= q:
            t = targets[cols.start - q: cols.stop - q]
        exact1 = (ids[:, 0] == t).float().mean()
        exact5 = (ids == t[:, None]).any(1).float().mean()
        # Same title in either collection counts as the right movie.
        hit_movie = [names[i].split("/")[1] if i >= 0 else "" for i in ids[:, 0].tolist()]
        true_movie = [names[i].split("/")[1] for i in t.tolist()]
        near = [h == m and abs(int(names[i][-10:-4]) - int(names[g][-10:-4])) <= 2
                for h, m, i, g in zip(hit_movie, true_movie, ids[:, 0].tolist(), t.tolist())]
        movie_ok = np.mean([h == m for h, m in zip(hit_movie, true_movie)])
        print(f"  {label:<28} {kind:<10} exact top-1 {exact1:6.1%}  top-5 {exact5:6.1%}  "
              f"within 2 frames {np.mean(near):6.1%}  right movie {movie_ok:6.1%}")

    real = args.queries  # query columns after this are real screenshots
    print(f"{args.queries} random frames, {len(names):,}-frame search")
    row("live site (whole frames)", live, slice(0, real), "crop")
    for name in methods:
        row(f"new: {name}", found[name], slice(0, real), "crop")
        row(f"new: {name} + pieces", found[f"{name} + pieces"], slice(0, real), "crop")
    row("live site (whole frames)", live, slice(q, q + real), "full frame")
    for name in methods:
        row(f"new: {name}", found[name], slice(q, q + real), "full frame")
    for n, spec in enumerate(args.extra):
        col = real + n
        print(f"\nreal screenshot -> {spec.split('=')[1]}")
        for label, top in [("live site", live)] + list(found.items()):
            ids = top.ids[col].tolist()
            rank = ids.index(int(targets[col])) + 1 if int(targets[col]) in ids else ">5"
            print(f"  {label:<28} top-1 {names[ids[0]]} ({float(top.scores[col, 0]):.3f}), target rank {rank}")
    note("done")


if __name__ == "__main__":
    main()
