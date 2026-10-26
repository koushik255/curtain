"""Curtain's encoder vs pretrained embedders (SSCD, DINOv2) on one film, on CPU.

Every frame of the film is embedded by each model (one whole-frame vector; the
current model also with its 49 shared-trunk tiles, as the site indexes it),
then synthetic queries of each validation-suite type, uploaded the way the
page uploads them, are ranked against all of the film's frames. Reports per
query type: exact frame at #1, same moment (within +-1 s, 2 fps) at #1, and
the median rank of the true frame. Frame vectors are cached under
.local/film-embedders/<film>-<frames>/.

    .venv/bin/python -m rcmindex.benchmark_film_embedders The_Godfather_Part_II_1974 --frames 6000
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from rcmindex import search
from rcmindex.compare_models import frame_vectors
from rcmindex.validation_suite import LOCAL, QUERY_TYPES, browser_upload, embed

FRAMES = LOCAL / "frames-small/l4-50k-lbfive"
CACHE = LOCAL / "film-embedders"
CURRENT = Path(__file__).resolve().parents[1] / "curtain_models/all-films-v1/best.pt"
# all-films-v1's index of the film, as rcmindex wrote it (49 tiles + .global).
CURRENT_RUNS = LOCAL / "rcmindex-runs-all-films-v1"
SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SAME_MOMENT = 2  # frames either side at 2 fps
BATCH = 32
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image.resize(size, Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


class Pretrained:
    """A pretrained embedder: images -> L2-normalized [n, d] float32."""

    def __init__(self, name: str, net: torch.nn.Module, size: callable):
        self.name, self.net, self.size = name, net.eval(), size

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        # Batch by size so non-square inputs keep their aspect ratio.
        out = [None] * len(images)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, image in enumerate(images):
            groups.setdefault(self.size(image), []).append(i)
        with torch.inference_mode():
            for size, members in groups.items():
                for start in range(0, len(members), BATCH):
                    chunk = members[start:start + BATCH]
                    v = self.net(torch.stack([tensor(images[i], size) for i in chunk]))
                    v = torch.nn.functional.normalize(v.float(), dim=-1).numpy()
                    for i, row in zip(chunk, v):
                        out[i] = row
        return np.stack(out)


def short_side(target: int, multiple: int = 1) -> callable:
    def size(image: Image.Image) -> tuple[int, int]:
        scale = target / min(image.size)
        return tuple(max(multiple, round(s * scale / multiple) * multiple) for s in image.size)
    return size


def sscd() -> Pretrained:
    path = LOCAL / "pretrained/sscd/sscd_disc_mixup.torchscript.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SSCD_URL, path)
    return Pretrained("sscd", torch.jit.load(str(path), map_location="cpu"), short_side(288))


def dinov2(variant: str) -> Pretrained:
    net = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{variant}14")
    return Pretrained(f"dinov2-{variant}", net, short_side(224, 14))


def embed_frames(model, paths: list[Path]) -> np.ndarray:
    pool = ThreadPoolExecutor(8)
    blocks = []
    for start in range(0, len(paths), 256):
        blocks.append(model(list(pool.map(load, paths[start:start + 256]))))
        if start % 2560 == 0:
            print(f"    {start + len(blocks[-1])}/{len(paths)}", flush=True)
    return np.concatenate(blocks).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("film", help="folder under .local/frames-small/l4-50k-lbfive")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--frames", type=int, default=None, help="only the film's first N frames (CPU time)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", default="current-tiles,current,sscd,dinov2-s")
    args = parser.parse_args()
    torch.set_num_threads(16)
    paths = sorted((FRAMES / args.film).glob("*.webp"))[:args.frames]
    names = [p.name for p in paths]
    print(f"{args.film}: {len(paths)} frames", flush=True)

    rng = random.Random(args.seed)
    items = []
    for kind, build in QUERY_TYPES.items():
        for _ in range(args.per_type):
            index = rng.randrange(len(paths))
            items.append((kind, browser_upload(build(load(paths[index]), rng)), index))
    print(f"{len(items)} queries", flush=True)

    current = CurtainEncoder.from_checkpoint(CURRENT, torch.device("cpu")).eval()
    report = {}
    for name in args.models.split(","):
        cache = CACHE / f"{args.film}-{len(paths)}" / f"{name}.npy"
        if name.startswith("current"):
            model = lambda images: embed(current, images)
        else:
            model = sscd() if name == "sscd" else dinov2(name.split("-")[1])
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            run = next(CURRENT_RUNS.glob(f"*/*/{args.film}.npy"), None)
            if name.startswith("current") and run is not None:
                records = json.loads(run.with_suffix(".records.json").read_text())
                assert [r.split("/")[-1][:-4] for r in records[:len(names)]] == [n[:-5] for n in names]
                whole = np.load(run.with_suffix(".global.npy"))[:len(names)]
                vectors = (np.concatenate([np.load(run)[:len(names)], whole[:, None]], axis=1)
                           if name == "current-tiles" else whole)
            elif name == "current-tiles":
                vectors = frame_vectors(current, paths, torch.device("cpu"))
            else:
                vectors = embed_frames(model, paths)
            np.save(cache, vectors)
            print(f"  {name}: frames embedded in {time.perf_counter() - started:.0f}s", flush=True)
        vectors = torch.from_numpy(np.load(cache)).float()
        rows = []
        started = time.perf_counter()
        for kind, image, target in items:
            if name == "current-tiles":
                # The site's rule: query + four corner pieces against 50 vectors per frame.
                q = embed(current, [image, *search.pieces(image)])
                q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
                best = (vectors @ q.T).amax(dim=1)  # [frames, 5]
                score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1))
            else:
                score = vectors @ torch.from_numpy(model([image])[0]).float()
            rank = int((score > score[target]).sum()) + 1
            top = int(score.argmax())
            rows.append((kind, rank, abs(top - target) <= SAME_MOMENT))
        print(f"  {name}: {len(items)} queries in {time.perf_counter() - started:.0f}s", flush=True)
        report[name] = rows

    kinds = list(QUERY_TYPES)
    models = list(report)
    print(f"\n{args.film}, {len(paths)} frames, {args.per_type} queries per type. "
          "Cells: exact #1 / same moment (+-1 s) #1 / median rank\n")
    print("| type | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for kind in [*kinds, "ALL"]:
        cells = []
        for m in models:
            sub = [r for r in report[m] if kind in ("ALL", r[0])]
            cells.append(f"{sum(r[1] == 1 for r in sub)} / {sum(r[2] for r in sub)} / "
                         f"{np.median([r[1] for r in sub]):.0f}")
        print(f"| {kind} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
