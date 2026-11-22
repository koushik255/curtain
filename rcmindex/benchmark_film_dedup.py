"""Merge near-duplicate vectors within shots: how many are kept vs accuracy, on one film, on CPU.

The index stores 50 vectors per frame (49 tiles + whole frame) at 2 fps, and
neighbouring frames of a shot are mostly the same picture. Two ways to drop the
repeats, each swept over cosine thresholds:

  frame   walk each shot in order; keep a frame when its whole-frame vector's
          cosine to the last kept frame's is below the threshold, else the
          frame is merged into it (all 50 of its vectors dropped)
  tile    the same per tile position (and for the whole-frame vector): a
          vector is kept when its cosine to the last kept vector at that
          position is below the threshold, so static background collapses
          while the parts that move keep their vectors

A merged vector is replaced by the kept one that covers it, which is what a
deduplicated index returns (each kept vector covers a frame range). Merging
compares against the last *kept* vector, not the previous frame, so drift
along a slow pan stays bounded by the threshold. Shots come from
rcmindex.shots (built if missing). "3 keyframes" is benchmark_film_shots'
fixed choice, for reference.

Queries of every validation-suite type, half from other films (half of those
from --hard), are scored against the film like search.search (query plus 4
pieces, whole-frame handicap). Reported per index: vectors kept, whether the
#1 result's range holds the target (frames whose vectors all merged into the
same kept ones are one result, a range), whether the time it shows (the
range's first frame) is within +-1 s of the target, right shot
#1, right shot among the top 5 shots, in-film AUC, and #1 hits for the crop
types alone, where tile vectors matter most.

    .venv/bin/python -m rcmindex.benchmark_film_dedup The_Godfather_Part_II_1974 --hard The_Godfather_1972
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import search
from rcmindex.benchmark_film_detection import auc
from rcmindex.benchmark_film_embedders import CURRENT, CURRENT_RUNS, FRAMES, load
from rcmindex.benchmark_film_shots import KEYFRAMES, SAME_MOMENT, keyframes, shot_map, top_shots
from rcmindex.validation_suite import QUERY_TYPES, browser_upload, embed

THRESHOLDS = [0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85]
CROPS = ["medium_crop", "tight_crop", "narrow_crop", "subtitles"]


def merge(vectors: np.ndarray, shot: np.ndarray, threshold: float) -> np.ndarray:
    """[frames, positions] index of the frame whose vector stands in at each position."""
    rep = np.empty(vectors.shape[:2], dtype=np.int64)
    last = np.zeros(vectors.shape[1], dtype=np.int64)
    for f in range(len(vectors)):
        if f == 0 or shot[f] != shot[f - 1]:
            last[:] = f
        else:
            cosine = np.einsum("pd,pd->p", vectors[f], vectors[last, np.arange(vectors.shape[1])])
            last = np.where(cosine < threshold, f, last)
        rep[f] = last
    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("film")
    parser.add_argument("--hard", required=True, help="look-alike film for half the negatives")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    torch.set_num_threads(16)
    rng = random.Random(args.seed)

    run = next(CURRENT_RUNS.glob(f"*/*/{args.film}.npy"))
    whole = np.load(run.with_suffix(".global.npy")).astype(np.float32)
    vectors = np.concatenate([np.load(run).astype(np.float32), whole[:, None]], axis=1)
    film = sorted((FRAMES / args.film).glob("*.webp"))
    assert len(film) == len(whole)
    frames, positions = vectors.shape[:2]
    shot = shot_map(args.film, whole)
    print(f"{args.film}: {frames} frames, {shot.max() + 1} shots, {frames * positions:,} vectors", flush=True)

    indexes = {"every frame": np.broadcast_to(np.arange(frames)[:, None], (frames, positions))}
    keep = keyframes(shot)
    nearest = np.empty(frames, dtype=np.int64)
    for s in np.unique(shot):
        members = np.flatnonzero(shot == s)
        keys = keep[shot[keep] == s]
        nearest[members] = keys[np.abs(members[:, None] - keys[None]).argmin(axis=1)]
    indexes[f"{KEYFRAMES} keyframes/shot"] = np.broadcast_to(nearest[:, None], (frames, positions))
    for t in THRESHOLDS:
        rep = merge(whole[:, None], shot, t)[:, 0]
        indexes[f"frame {t:.2f}"] = np.broadcast_to(rep[:, None], (frames, positions))
    for t in THRESHOLDS:
        indexes[f"tile {t:.2f}"] = merge(vectors, shot, t)
        print(f"  merged tiles at {t:.2f}", flush=True)
    kept = {name: int((rep == np.arange(frames)[:, None]).sum()) for name, rep in indexes.items()}

    hard = sorted((FRAMES / args.hard).glob("*.webp"))
    others = sorted(p for p in FRAMES.iterdir() if p.is_dir() and p.name not in (args.film, args.hard))
    items = []  # (type, target frame or -1, image)
    for kind in QUERY_TYPES:
        for n in range(args.per_type):
            target = rng.randrange(frames)
            items.append((kind, target, browser_upload(QUERY_TYPES[kind](load(film[target]), rng))))
            source = rng.choice(hard) if n % 2 else rng.choice(sorted(rng.choice(others).glob("*.webp")))
            items.append((kind, -1, browser_upload(QUERY_TYPES[kind](load(source), rng))))
    print(f"{len(items)} queries, half from other films", flush=True)

    model = CurtainEncoder.from_checkpoint(CURRENT, torch.device("cpu")).eval()
    table = torch.from_numpy(vectors)
    rows = []  # per query: type, target, then per index: (shown frame, best score, top-5 shots, #1 in range)
    for number, (kind, target, image) in enumerate(items, 1):
        q = embed(model, [image, *search.pieces(image)])
        q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
        sims = (table @ q.T).numpy()  # [frames, positions, queries]
        row = [kind, target]
        for rep in indexes.values():
            best = sims[rep, np.arange(positions)].max(axis=1)
            score = np.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(axis=1))
            top = shown = int(np.argmax(score))
            # The index can't tell apart frames whose vectors all merged into the same kept ones: it
            # returns their range and shows its first frame's time.
            while shown > 0 and (rep[shown - 1] == rep[top]).all():
                shown -= 1
            row.append((shown, float(score[top]), top_shots(score, np.arange(frames), shot, 5),
                        target >= 0 and bool((rep[top] == rep[target]).all())))
        rows.append(row)
        if number % 50 == 0:
            print(f"  {number}/{len(items)}", flush=True)

    pos = [r for r in rows if r[1] >= 0]
    crops = [r for r in pos if r[0] in CROPS]
    print(f"\n{len(pos)} queries from the film ({len(crops)} crops), {len(rows) - len(pos)} from others\n")
    print("| index | vectors kept | #1 range has target | #1 shown time within +-1 s | right shot #1 | right shot in top 5 shots "
          "| in-film AUC | crops: #1 range has target | crops: right shot #1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for column, name in enumerate(indexes, 2):
        def hit(r):
            return r[column][3]
        moment = sum(abs(r[column][0] - r[1]) <= SAME_MOMENT for r in pos)
        right = sum(shot[r[column][0]] == shot[r[1]] for r in pos)
        in5 = sum(shot[r[1]] in r[column][2] for r in pos)
        p = np.array([r[column][1] for r in pos])
        n = np.array([r[column][1] for r in rows if r[1] < 0])
        print(f"| {name} | {kept[name]:,} ({kept[name] / (frames * positions):.0%}) | {sum(map(hit, pos))} "
              f"| {moment} | {right} | {in5} | {auc(p, n):.3f} | {sum(map(hit, crops))}/{len(crops)} "
              f"| {sum(shot[r[column][0]] == shot[r[1]] for r in crops)} |")


if __name__ == "__main__":
    main()
