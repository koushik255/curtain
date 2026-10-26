"""Is this screenshot from the film or not? Detection on one film, on CPU.

Positives are queries of every validation-suite type made from the film's
first --frames frames; negatives are the same query types made from other
films: half from --hard (a look-alike film, e.g. its sequel or prequel), half
one frame each from other films. Each query's score is its best match among
the film's frames (the site's rule for the tiled model, cosine otherwise),
using the frame vectors benchmark_film_embedders cached. Reports per model:
ROC AUC, the share of positives caught at 1% and 5% false positives, and the
best accuracy over all thresholds, overall, per negative group and per type.

    .venv/bin/python -m rcmindex.benchmark_film_detection The_Godfather_Part_II_1974 \\
        --frames 6000 --hard The_Godfather_1972
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from curtain_ml.model import CurtainEncoder
from rcmindex import search
from rcmindex.benchmark_film_embedders import CACHE, CURRENT, FRAMES, dinov2, load, sscd
from rcmindex.validation_suite import QUERY_TYPES, browser_upload, embed


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    # Probability a random positive outscores a random negative (ties count half).
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def tpr_at(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    threshold = np.quantile(neg, 1 - fpr)
    return float((pos > threshold).mean())


def best_accuracy(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    candidates = np.concatenate([pos, neg])
    accuracy = [((pos >= t).sum() + (neg < t).sum()) / (len(pos) + len(neg)) for t in candidates]
    i = int(np.argmax(accuracy))
    return float(accuracy[i]), float(candidates[i])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("film")
    parser.add_argument("--frames", type=int, default=6000, help="must match a benchmark_film_embedders cache")
    parser.add_argument("--hard", required=True, help="look-alike film for half the negatives")
    parser.add_argument("--per-type", type=int, default=20, help="positives per type; as many negatives")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--models", default="current-tiles,current,sscd,dinov2-s")
    args = parser.parse_args()
    torch.set_num_threads(16)
    rng = random.Random(args.seed)

    film = sorted((FRAMES / args.film).glob("*.webp"))[:args.frames]
    hard = sorted((FRAMES / args.hard).glob("*.webp"))
    others = sorted(p for p in FRAMES.iterdir() if p.is_dir() and p.name not in (args.film, args.hard))
    items = []  # (type, group, image)
    for kind, build in QUERY_TYPES.items():
        for n in range(args.per_type):
            items.append((kind, "film", browser_upload(build(load(rng.choice(film)), rng))))
            if n % 2:
                source = rng.choice(hard)
                group = "hard"
            else:
                source = rng.choice(sorted(rng.choice(others).glob("*.webp")))
                group = "other"
            items.append((kind, group, browser_upload(build(load(source), rng))))
    print(f"{args.film} ({len(film)} frames): {len(items)} queries, half from the film", flush=True)

    current = CurtainEncoder.from_checkpoint(CURRENT, torch.device("cpu")).eval()
    scores = {}
    for name in args.models.split(","):
        vectors = torch.from_numpy(np.load(CACHE / f"{args.film}-{len(film)}" / f"{name}.npy")).float()
        if not name.startswith("current"):
            model = sscd() if name == "sscd" else dinov2(name.split("-")[1])
        out = []
        for _, _, image in items:
            if name == "current-tiles":
                q = embed(current, [image, *search.pieces(image)])
                q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
                best = (vectors @ q.T).amax(dim=1)
                score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1))
            elif name == "current":
                q = embed(current, [image])[0]
                score = vectors @ torch.from_numpy(q / np.linalg.norm(q)).float()
            else:
                score = vectors @ torch.from_numpy(model([image])[0]).float()
            out.append(float(score.max()))
        scores[name] = np.array(out)
        print(f"  {name} scored", flush=True)

    kinds = np.array([k for k, _, _ in items])
    groups = np.array([g for _, g, _ in items])
    np.savez(CACHE / f"{args.film}-{len(film)}" / "detection-scores.npz", kinds=kinds, groups=groups, **scores)

    def row(label: str, pos_mask: np.ndarray, neg_mask: np.ndarray) -> None:
        cells = []
        for name, s in scores.items():
            pos, neg = s[pos_mask], s[neg_mask]
            accuracy, _ = best_accuracy(pos, neg)
            cells.append(f"{auc(pos, neg):.3f} / {tpr_at(pos, neg, 0.01):.0%} / {tpr_at(pos, neg, 0.05):.0%} / {accuracy:.0%}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print(f"\nCells: AUC / caught at 1% false positives / caught at 5% / best accuracy\n")
    print("| negatives | " + " | ".join(scores) + " |")
    print("|---|" + "---|" * len(scores))
    film_mask = groups == "film"
    row("all", film_mask, ~film_mask)
    row(f"{args.hard} only", film_mask, groups == "hard")
    row("other films only", film_mask, groups == "other")
    print("\n| type (all negatives) | " + " | ".join(scores) + " |")
    print("|---|" + "---|" * len(scores))
    for kind in QUERY_TYPES:
        row(kind, film_mask & (kinds == kind), ~film_mask & (kinds == kind))
    for name, s in scores.items():
        _, threshold = best_accuracy(s[film_mask], s[~film_mask])
        print(f"{name}: best threshold {threshold:.3f}; film median {np.median(s[film_mask]):.3f}, "
              f"{args.hard} median {np.median(s[groups == 'hard']):.3f}, "
              f"other median {np.median(s[groups == 'other']):.3f}")


if __name__ == "__main__":
    main()
