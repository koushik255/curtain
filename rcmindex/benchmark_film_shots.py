"""Search by shot instead of by frame, on one film, with the live model, on CPU.

Shots: rcmindex.shots' map of the film (PySceneDetect cuts confirmed by the
live model's whole-frame vectors), built if missing.

Queries of the bigger validation-suite types (no tight crops or slivers) are
searched against all of the film's frames with all-films-v1's index vectors
and the site's scoring rule, then the hits are grouped by shot. Reports: exact
frame and same moment at #1 (as now), right shot at #1, right shot among the
top 5 frames vs the top 5 shots, the same with only 3 keyframes kept per shot,
and how well the best score tells the film's screenshots from as many from
other films (half from --hard).

    .venv/bin/python -m rcmindex.benchmark_film_shots The_Godfather_Part_II_1974 --hard The_Godfather_1972
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import search, shots
from rcmindex.benchmark_film_detection import auc, best_accuracy, tpr_at
from rcmindex.benchmark_film_embedders import CURRENT, CURRENT_RUNS, FRAMES, load
from rcmindex.validation_suite import QUERY_TYPES, browser_upload, embed

BIG = [k for k in QUERY_TYPES if k not in ("tight_crop", "narrow_crop")]
KEYFRAMES = 3
SAME_MOMENT = 2  # frames either side at 2 fps


def shot_map(film: str, whole: np.ndarray) -> np.ndarray:
    """Shot number of every frame, from rcmindex.shots' map (built here if missing)."""
    movie = f"{FRAMES.name}/{film}"
    if (shot := shots.load(movie, len(whole))) is None:
        cuts, detected = shots.detect(shots.frame_pattern(movie), whole)
        shots.path(movie).parent.mkdir(parents=True, exist_ok=True)
        shots.path(movie).write_text(json.dumps({"fps": 2, "frames": len(whole), "cuts": cuts,
                                                 "detected": detected, "cut": shots.CUT}))
        shot = shots.load(movie, len(whole))
    return shot


def keyframes(shot: np.ndarray) -> np.ndarray:
    keep = []
    for s in np.unique(shot):
        members = np.flatnonzero(shot == s)
        if len(members) <= KEYFRAMES:
            keep += list(members)
        else:
            keep += [members[len(members) * (2 * k + 1) // (2 * KEYFRAMES)] for k in range(KEYFRAMES)]
    return np.array(keep)


def top_shots(score: np.ndarray, frames: np.ndarray, shot: np.ndarray, n: int) -> list[int]:
    out = []
    for f in frames[np.argsort(-score)]:
        if shot[f] not in out:
            out.append(int(shot[f]))
            if len(out) == n:
                break
    return out


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
    vectors = torch.from_numpy(np.concatenate([np.load(run), whole[:, None].astype(np.float16)], axis=1)).float()
    film = sorted((FRAMES / args.film).glob("*.webp"))
    assert len(film) == len(whole)
    shot = shot_map(args.film, whole)
    keep = keyframes(shot)
    lengths = np.bincount(shot)
    print(f"{args.film}: {len(film)} frames, {lengths.size} shots (median {np.median(lengths) / 2:.1f} s, "
          f"mean {lengths.mean() / 2:.1f} s), {len(keep)} keyframes ({len(keep) / len(film):.0%})", flush=True)

    hard = sorted((FRAMES / args.hard).glob("*.webp"))
    others = sorted(p for p in FRAMES.iterdir() if p.is_dir() and p.name not in (args.film, args.hard))
    items = []  # (type, target frame or -1, image)
    for kind in BIG:
        for n in range(args.per_type):
            target = rng.randrange(len(film))
            items.append((kind, target, browser_upload(QUERY_TYPES[kind](load(film[target]), rng))))
            source = rng.choice(hard) if n % 2 else rng.choice(sorted(rng.choice(others).glob("*.webp")))
            items.append((kind, -1, browser_upload(QUERY_TYPES[kind](load(source), rng))))
    print(f"{len(items)} queries, half from other films", flush=True)

    model = CurtainEncoder.from_checkpoint(CURRENT, torch.device("cpu")).eval()
    every = np.arange(len(film))
    rows = []  # per query: type, target, then per index (every, keyframes): top frame, best score, top-5 frames, top-5 shots
    for number, (kind, target, image) in enumerate(items, 1):
        q = embed(model, [image, *search.pieces(image)])
        q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
        best = (vectors @ q.T).amax(dim=1)
        score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1)).numpy()
        row = [kind, target]
        for frames in (every, keep):
            s = score[frames]
            order = frames[np.argsort(-s)]
            row.append((int(order[0]), float(s.max()), order[:5], top_shots(s, frames, shot, 5)))
        rows.append(row)
        if number % 50 == 0:
            print(f"  {number}/{len(items)}", flush=True)

    def table(label: str, column: int) -> None:
        print(f"\n{label}\n")
        print("| type | exact frame #1 | same moment (+-1 s) #1 | right shot #1 | right shot in top 5 frames "
              "| right shot in top 5 shots | in-film AUC | caught at 5% false positives | best accuracy |")
        print("|---|---|---|---|---|---|---|---|---|")
        for kind in [*BIG, "ALL"]:
            mine = [r for r in rows if kind in ("ALL", r[0])]
            pos = [r for r in mine if r[1] >= 0]
            top = np.array([r[column][0] for r in pos])
            target = np.array([r[1] for r in pos])
            right = shot[top] == shot[target]
            in5_frames = sum(shot[t] in shot[r[column][2]] for t, r in zip(target, pos))
            in5_shots = sum(shot[t] in r[column][3] for t, r in zip(target, pos))
            p = np.array([r[column][1] for r in pos])
            n = np.array([r[column][1] for r in mine if r[1] < 0])
            print(f"| {kind} | {(top == target).sum()}/{len(pos)} | {(abs(top - target) <= SAME_MOMENT).sum()} "
                  f"| {right.sum()} | {in5_frames} | {in5_shots} | {auc(p, n):.3f} | {tpr_at(p, n, 0.05):.0%} "
                  f"| {best_accuracy(p, n)[0]:.0%} |")

    table(f"Index: every frame ({len(film)})", 2)
    table(f"Index: {KEYFRAMES} keyframes per shot ({len(keep)})", 3)


if __name__ == "__main__":
    main()
