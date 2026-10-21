"""Old vs new encoder, without re-indexing: rank of the true frame within its film.

For each checkpoint, every frame of the six validation movies is embedded the
way the index is (rcmindex's shared trunk: 49 tiles + the whole frame), and
each query (the validation suite plus --extra fresh ones per type, as the page
uploads them) is scored against all frames of its movie with search.rescore's
rule. Reports, per query type: how often the true frame ranks first among its
movie's ~15-20k frames, the median rank, and the true frame's mean score.
Needs a GPU; the frame embeddings are cached per checkpoint under
.local/compare-models/.

    .venv/bin/python -m rcmindex.compare_models OLD.pt NEW.pt --extra 30
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import IMAGENET_MEAN, IMAGENET_STD
from rcmindex import search
from rcmindex.benchmark_int8 import queries
from rcmindex.export_model import GRID, STEPS, trunk_of
from rcmindex.validation_suite import LOCAL, SOURCES, VALIDATION, browser_upload, embed

CACHE = LOCAL / "compare-models"
BATCH = 16


def normalized(frames: torch.Tensor, size: int) -> torch.Tensor:
    # export_model.normalized, on the frames' device.
    images = torch.nn.functional.interpolate(
        frames.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frames.device).view(1, 3, 1, 1)
    return (images / 255 - mean) / std


def frame_vectors(model: CurtainEncoder, paths: list[Path], device: torch.device) -> np.ndarray:
    """[frames, 50, 128] fp16: 49 shared-trunk tiles then the whole frame."""
    trunk = trunk_of(model).to(device).eval()
    out = np.empty((len(paths), 50, 128), dtype=np.float16)
    pool = ThreadPoolExecutor(8)

    def load(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(paths), BATCH):
            frames = torch.stack(list(pool.map(load, paths[start:start + BATCH]))).to(device)
            features = trunk(normalized(frames, GRID * model.image_size))  # [b, 512, 24, 24]
            cells = features.shape[-1] // GRID
            stride = cells // 2
            pooled = torch.stack([
                features[:, :, y * stride:y * stride + cells, x * stride:x * stride + cells].mean((2, 3))
                for y in range(STEPS) for x in range(STEPS)
            ], dim=1)  # [b, 49, 512]
            tiles = torch.nn.functional.normalize(model.backbone.fc(pooled.float()), dim=-1)
            whole = model(normalized(frames, model.image_size))
            out[start:start + len(frames), :49] = tiles.cpu().numpy()
            out[start:start + len(frames), 49] = whole.float().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--extra", type=int, default=30, help="fresh queries per type on top of the suite")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    device = torch.device("cuda")
    items = queries(args.extra, args.seed)
    print(f"{len(items)} queries", flush=True)
    paths = {m: sorted((SOURCES["full-curtain"] / m).glob("*.jpg")) for m in VALIDATION}
    report = {}
    for checkpoint in args.checkpoints:
        model = CurtainEncoder.from_checkpoint(checkpoint, device).eval()
        tag = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]
        vectors = {}
        for movie in VALIDATION:
            cache = CACHE / tag / f"{movie}.npy"
            if not cache.exists():
                started = time.perf_counter()
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, frame_vectors(model, paths[movie], device))
                print(f"  {checkpoint.parent.name}: {movie} embedded in {time.perf_counter() - started:.0f}s", flush=True)
            vectors[movie] = torch.from_numpy(np.load(cache)).to(device)
        rows = []
        model = model.cpu()
        for kind, image, target in items:
            movie, name = target.split("/")[1:]
            image = browser_upload(image)
            q = embed(model, [image, *search.pieces(image)])
            q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True)).to(device).half()
            best = (vectors[movie] @ q.T).amax(dim=1).float()  # [frames, q]
            score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1))
            index = [p.name for p in paths[movie]].index(name)
            rank = int((score > score[index]).sum()) + 1
            rows.append((kind, rank, float(score[index])))
        model = model.to(device)
        report[checkpoint] = rows

    kinds = list(dict.fromkeys(kind for kind, _, _ in items))
    names = [c.parent.name for c in args.checkpoints]
    print("\n| type | " + " | ".join(f"{n} #1" for n in names) + " | " + " | ".join(f"{n} median rank" for n in names)
          + " | " + " | ".join(f"{n} true score" for n in names) + " |")
    print("|---|" + "---|" * (3 * len(names)))
    for kind in [*kinds, "ALL"]:
        cells = []
        for metric in ("first", "rank", "score"):
            for c in args.checkpoints:
                sub = [r for r in report[c] if kind in ("ALL", r[0])]
                if metric == "first":
                    cells.append(f"{sum(r[1] == 1 for r in sub)}/{len(sub)}")
                elif metric == "rank":
                    cells.append(f"{np.median([r[1] for r in sub]):.0f}")
                else:
                    cells.append(f"{np.mean([r[2] for r in sub]):.3f}")
        print(f"| {kind} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
