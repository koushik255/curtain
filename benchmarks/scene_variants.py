"""Which scene index ranks the right film best? (curtain_ml/scene_modal.py::dense)

For each DINOv2 size's dense index (every 8th frame), on this machine:
embeds the benchmark queries (as scene_eval's: web-search-v2 misses and every
online-v1 query), then ranks films under each variant:

    density   every 8th / 16th / 32nd frame of the index (thinned locally)
    score     a film's best frame ("max") or the mean of its best k frames

and reports the scene films alone and merged after the site's confirmed
films (scene_eval's scene2nd), right film counted through film_aliases.

    .venv/bin/python -m benchmarks.scene_variants --index .local/scene-index/dense-8 --sizes small,base,large
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmarks.scene_eval import B, films, merge
from curtain_ml.scene_modal import BACKBONES
from rcmindex.film_aliases import canonical

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cases() -> list[tuple[str, str, Path, dict]]:
    """(group, key, query image, the site's stored result)."""
    reviews = json.loads((B / "web-search-v2/reviews.json").read_text())
    v2 = {json.loads(l)["id"]: json.loads(l) for l in (B / "web-search-v2/results.jsonl").read_text().splitlines()
          if json.loads(l)["variant"] == "original"}
    v1 = {json.loads(l)["id"]: json.loads(l) for l in (B / "online-v1-live/results.jsonl").read_text().splitlines()}
    out = [("web-search-v2 " + reviews[p.name[:-6]], "v2/" + p.name[:-6], p, v2[p.name[:-6]])
           for p in sorted((B / "web-search-v2/misses-site/img").glob("*.q.jpg")) if reviews.get(p.name[:-6]) != "exclude"]
    out += [("online-v1", "v1/" + i, B / "online-v1/queries" / f"{i}.jpg", r) for i, r in sorted(v1.items())]
    return out


@torch.inference_mode()
def embed_queries(size: str, paths: list[Path]) -> np.ndarray:
    import timm
    model = timm.create_model(BACKBONES[size], pretrained=True, num_classes=0, img_size=SIZE).eval()
    out = []
    for s in range(0, len(paths), 32):
        batch = []
        for p in paths[s:s + 32]:
            with Image.open(p) as im:
                small = im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0)
            batch.append((np.asarray(small, dtype=np.float32) / 255 - MEAN) / STD)
        v = model(torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)).numpy()
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out).astype(np.float32)


def rank(queries: torch.Tensor, bank: torch.Tensor, film_of: np.ndarray, n_films: int, ks: tuple) -> dict:
    """{k: [queries, films] score}: the mean of each film's best k frames."""
    starts = np.searchsorted(film_of, np.arange(n_films + 1))
    longest = int(np.diff(starts).max())
    grid = np.full((n_films, longest), len(film_of), dtype=np.int64)  # padding -> a -inf column
    for f in range(n_films):
        grid[f, :starts[f + 1] - starts[f]] = np.arange(starts[f], starts[f + 1])
    grid = torch.from_numpy(grid)
    out = {k: [] for k in ks}
    for s in range(0, len(queries), 16):
        sims = queries[s:s + 16] @ bank.T
        sims = torch.cat([sims, torch.full((len(sims), 1), float("-inf"))], 1)
        top = sims[:, grid].topk(max(ks), dim=2).values  # [q, films, k]
        for k in ks:
            out[k].append(top[:, :, :k].clamp(min=-1).mean(2))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


POLICIES = ("scene2nd", "weak30", "weak40", "weak60")


def merged(policy: str, site: list[dict], scene: list[str]) -> list[str]:
    """scene2nd (scene_eval), or weak<T>: as scene2nd, but when the top
    confirmed match has under T points and isn't in the scene's top 3, the
    scene's #1 goes first and that match second."""
    shown = merge("scene2nd", site, scene)
    if policy == "scene2nd":
        return shown
    confirmed = [x for x in site if x.get("confirmed")]
    if confirmed and confirmed[0]["points"] < int(policy[4:]) and confirmed[0]["movie"] not in scene[:3]:
        shown = [scene[0]] + [f for f in shown if f != scene[0]]
    return shown[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sizes", default="small,base,large")
    args = parser.parse_args()
    all_cases = cases()
    print("| size | frames | score | policy | set | n | scene #1 | scene top 20 | merged #1 | merged top 5 | merged top 20 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for size in args.sizes.split(","):
        frames = json.loads((args.index / size / "frames.json").read_text())
        vectors = np.load(args.index / size / "vectors.npy", mmap_mode="r")
        names = sorted({f for f, _ in frames})
        film_of_all = np.searchsorted(np.array(names), np.array([f for f, _ in frames]))
        starts = np.searchsorted(film_of_all, np.arange(len(names)))
        position = np.arange(len(frames)) - starts[film_of_all]
        queries = embed_queries(size, [c[2] for c in all_cases])
        for thin in (1, 2, 4):
            keep = np.nonzero(position % thin == 0)[0]
            bank = torch.from_numpy(np.asarray(vectors[keep], dtype=np.float32))
            film_of = film_of_all[keep]
            ranked = rank(torch.from_numpy(queries), bank, film_of, len(names), (1, 3))
            for k, scores in ranked.items():
                for policy in POLICIES:
                    tally = defaultdict(lambda: [0] * 6)
                    for (group, key, _, r), row in zip(all_cases, scores):
                        scene = list(dict.fromkeys(canonical(names[f]) for f in np.argsort(-row)[:60]))
                        site = [dict(x, movie=canonical(x["movie"])) for x in r["results"]]
                        shown = merged(policy, site, scene)
                        target = canonical(r["movie"])
                        t = tally[group]
                        t[0] += 1
                        t[1] += scene[:1] == [target]; t[2] += target in scene[:20]
                        t[3] += shown[:1] == [target]; t[4] += target in shown[:5]; t[5] += target in shown[:20]
                    for g in sorted(tally):
                        n, a, b, c, d, e = tally[g]
                        print(f"| {size} | every {8 * thin} | {'max' if k == 1 else f'top-{k} mean'} | {policy} | {g} | {n} "
                              f"| {a} | {b} | {c} | {d} | {e} |", flush=True)

if __name__ == "__main__":
    main()
