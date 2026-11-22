"""Which tiles of a frame are worth indexing? One film, on CPU.

Each frame is embedded as the site indexes it (49 shared-trunk tiles, each a
quarter of a side, + the whole frame) and also with a finer grid (225 tiles,
each an eighth of a side, from one trunk pass at twice the size). Every tile
gets a detail score from its pixels (mean gradient magnitude of the grey
frame), so flat regions (sky, walls, black bars) can be left out.

Queries of the crop types (and a "tiny_crop", 10-18% of each side) are ranked
against all the frames with the site's rule under each tile set:

  current        49 coarse + whole (what the site has)
  drop-flat P    coarse tiles below the P-th percentile of detail left out
  drop-random P  the same number of coarse tiles left out at random (control)
  +fine          coarse + all 225 fine tiles (the ceiling, 5.5x the vectors)
  budget         flat coarse tiles out, the most detailed fine tiles in, to
                 the same 50 vectors per frame on average as today
  budget-random  the same counts, tiles picked at random (control)
  detailed       every coarse and fine tile that is not flat, whatever it adds up to

Reports exact frame #1 / same moment #1 / median rank per query type, for all
queries and for the ones cut from a detailed region. Within one film only: it
does not measure false matches against the other films. Frame vectors are
cached under .local/film-tiles/.

    .venv/bin/python -m rcmindex.benchmark_film_tiles The_Godfather_Part_II_1974 --start 1500 --frames 2500
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw

from curtain_ml.model import CurtainEncoder
from rcmindex import search
from rcmindex.export_model import GRID, normalized, trunk_of
from rcmindex.validation_suite import LOCAL, QUERY_TYPES, browser_upload, crop, embed, jpeg
from rcmindex.data_paths import WEBP

FRAMES = WEBP / "l4-50k-lbfive"
CACHE = LOCAL / "film-tiles"
CURRENT = Path(__file__).resolve().parents[1] / "curtain_models/all-films-v1/best.pt"
SAME_MOMENT = 2  # frames either side at 2 fps
BATCH = 4
COARSE, FINE = 49, 225
# Grey frames are resized to this for the detail scores, so both grids' tiles
# are whole numbers of pixels: coarse 80x160 at steps of 40x80, fine half that.
DETAIL_SIZE = (320, 640)
KINDS = {
    "full_frame": QUERY_TYPES["full_frame"],
    "wide_crop": QUERY_TYPES["wide_crop"],
    "medium_crop": QUERY_TYPES["medium_crop"],
    "subtitles": QUERY_TYPES["subtitles"],
    "tight_crop": QUERY_TYPES["tight_crop"],
    "narrow_crop": QUERY_TYPES["narrow_crop"],
    "tiny_crop": lambda f, r: jpeg(crop(f, r, (0.10, 0.18), (0.10, 0.18)), 85),
}


def load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def gradient(grey: torch.Tensor) -> torch.Tensor:
    """[b, 1, h, w] grey 0-255 -> gradient magnitude, same shape."""
    dx = functional.pad(grey[..., :, 1:] - grey[..., :, :-1], (0, 1))
    dy = functional.pad(grey[..., 1:, :] - grey[..., :-1, :], (0, 0, 0, 1))
    return (dx ** 2 + dy ** 2).sqrt()


def detail_of(image: Image.Image) -> float:
    """A whole image's detail score, on the scale of tile_detail."""
    grey = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32))[None, None]
    h, w = grey.shape[-2:]
    # Blur away what the frames' own 640 -> 320 high shrink would.
    grey = functional.interpolate(grey, size=(max(8, h // 2), max(8, w)), mode="bilinear", antialias=True)
    return float(gradient(grey).mean())


def tile_detail(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [b, 3, h, w] -> detail of the 49 coarse and the 225 fine tiles."""
    grey = (frames.float() * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    grey = functional.interpolate(grey, size=DETAIL_SIZE, mode="bilinear", antialias=True)
    g = gradient(grey)
    h, w = DETAIL_SIZE
    coarse = functional.avg_pool2d(g, (h // 4, w // 4), (h // 8, w // 8))
    fine = functional.avg_pool2d(g, (h // 8, w // 8), (h // 16, w // 16))
    return coarse.flatten(1), fine.flatten(1)


def embed_film(model: CurtainEncoder, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """([frames, 49 + 225 + 1, 128] fp16, [frames, 49 + 225] detail)."""
    trunk = trunk_of(model).eval().to(memory_format=torch.channels_last)
    vectors = np.empty((len(paths), COARSE + FINE + 1, 128), dtype=np.float16)
    detail = np.empty((len(paths), COARSE + FINE), dtype=np.float32)
    pool = ThreadPoolExecutor(8)
    started = time.perf_counter()

    def tiles(frames: torch.Tensor, scale: int) -> torch.Tensor:
        features = trunk(normalized(frames, scale * GRID * model.image_size).contiguous(memory_format=torch.channels_last))
        cells = features.shape[-1] // (scale * GRID)  # 6 feature cells per tile
        pooled = functional.avg_pool2d(features, cells, cells // 2).flatten(2).transpose(1, 2)
        return functional.normalize(model.backbone.fc(pooled), dim=-1)

    with torch.inference_mode():
        for start in range(0, len(paths), BATCH):
            images = list(pool.map(load, paths[start:start + BATCH]))
            frames = torch.stack([torch.from_numpy(np.asarray(i).copy()).permute(2, 0, 1) for i in images])
            rows = slice(start, start + len(frames))
            vectors[rows, :COARSE] = tiles(frames, 1).numpy()
            vectors[rows, COARSE:COARSE + FINE] = tiles(frames, 2).numpy()
            vectors[rows, -1] = model(normalized(frames, model.image_size)).numpy()
            detail[rows] = torch.cat(tile_detail(frames), dim=1).numpy()
            if start % 200 == 0:
                print(f"    {start + len(frames)}/{len(paths)}  {time.perf_counter() - started:.0f}s", flush=True)
    return vectors, detail


def overlay(path: Path, keep: np.ndarray) -> Image.Image:
    """The frame with the fine tiles that are left out darkened."""
    image = load(path)
    w, h = image.size
    covered = np.zeros((16, 16), dtype=bool)  # sixteenths of each side covered by a kept fine tile
    for i in np.flatnonzero(keep):
        y, x = divmod(int(i), 15)
        covered[y:y + 2, x:x + 2] = True
    shade = Image.new("RGB", image.size, (200, 0, 60))
    mask = Image.fromarray((~covered).astype(np.uint8) * 150).resize((w, h), Image.NEAREST)
    return Image.composite(shade, image, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("film", help="folder under .local/data/frames/webp/l4-50k-lbfive")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=2500, help="this many consecutive frames (CPU time)")
    parser.add_argument("--per-type", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--percentiles", default="20,40,60", help="coarse detail percentiles to drop below")
    args = parser.parse_args()
    torch.set_num_threads(16)
    paths = sorted((FRAMES / args.film).glob("*.webp"))[args.start:args.start + args.frames]
    print(f"{args.film}: frames {args.start}..{args.start + len(paths)}", flush=True)
    model = CurtainEncoder.from_checkpoint(CURRENT, torch.device("cpu")).eval()

    cache = CACHE / f"{args.film}-{args.start}-{len(paths)}"
    if not (cache / "vectors.npy").exists():
        cache.mkdir(parents=True, exist_ok=True)
        vectors, detail = embed_film(model, paths)
        np.save(cache / "detail.npy", detail)
        np.save(cache / "vectors.npy", vectors)
    vectors = torch.from_numpy(np.load(cache / "vectors.npy")).float()  # [F, 275, 128]
    detail = np.load(cache / "detail.npy")
    coarse_detail, fine_detail = detail[:, :COARSE], detail[:, COARSE:]
    count = len(paths)

    # Tile sets: [F, 275] bool, True = indexed. The whole-frame vector always is.
    rng = np.random.default_rng(args.seed)

    def tile_set(coarse: np.ndarray, fine: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([coarse, fine, np.ones((count, 1), dtype=bool)], axis=1))

    def at_random(shape: tuple[int, int], kept: int) -> np.ndarray:
        flat = np.zeros(shape[0] * shape[1], dtype=bool)
        flat[rng.choice(flat.size, kept, replace=False)] = True
        return flat.reshape(shape)

    no_fine = np.zeros((count, FINE), dtype=bool)
    sets = {"current": tile_set(np.ones((count, COARSE), dtype=bool), no_fine)}
    cuts = {}
    for p in [int(x) for x in args.percentiles.split(",")]:
        cuts[p] = float(np.percentile(coarse_detail, p))
        keep = coarse_detail >= cuts[p]
        sets[f"drop-flat {p}"] = tile_set(keep, no_fine)
        sets[f"drop-random {p}"] = tile_set(at_random(keep.shape, int(keep.sum())), no_fine)
    sets["+fine"] = tile_set(np.ones((count, COARSE), dtype=bool), np.ones((count, FINE), dtype=bool))
    # Same size as today: flat coarse tiles out, the most detailed fine tiles in.
    middle = sorted(cuts)[len(cuts) // 2]
    keep = coarse_detail >= cuts[middle]
    room = count * COARSE - int(keep.sum())
    fine_cut = np.sort(fine_detail, axis=None)[-room]
    sets["budget"] = tile_set(keep, fine_detail >= fine_cut)
    sets["budget-random"] = tile_set(at_random(keep.shape, int(keep.sum())), at_random(fine_detail.shape, room))
    # No size limit: every tile of either grid that is not flat.
    sets["detailed"] = tile_set(keep, fine_detail >= cuts[middle])
    print(f"coarse detail cuts {cuts}; budget = drop-flat {middle} + fine tiles with detail >= {fine_cut:.2f}")

    # A picture of what the budget set leaves out, on evenly spaced frames.
    picks = np.linspace(0, count - 1, 12).astype(int)
    sheet = Image.new("RGB", (4 * 640, 3 * 360))
    for n, i in enumerate(picks):
        covered = sets["budget"][i, COARSE:COARSE + FINE].numpy().copy()
        for c in np.flatnonzero(sets["budget"][i, :COARSE].numpy()):  # a coarse tile covers 3x3 fine positions
            y, x = divmod(int(c), 7)
            covered.reshape(15, 15)[2 * y:2 * y + 3, 2 * x:2 * x + 3] = True
        tile = overlay(paths[i], covered).resize((640, 360))
        ImageDraw.Draw(tile).text((6, 4), paths[i].stem, fill=(255, 255, 255))
        sheet.paste(tile, (n % 4 * 640, n // 4 * 360))
    sheet.save(cache / "budget-tiles.jpg", quality=88)
    print(f"wrote {cache / 'budget-tiles.jpg'} (red = no tile there)")

    random_ = random.Random(args.seed)
    items = []
    for kind, build in KINDS.items():
        for _ in range(args.per_type):
            index = random_.randrange(count)
            query = build(load(paths[index]), random_)
            items.append((kind, browser_upload(query), index, detail_of(query)))
    # A query is "detailed" if a coarse tile like it would survive the middle cut.
    detailed_cut = cuts[middle]
    print(f"{len(items)} queries, {sum(d >= detailed_cut for *_, d in items)} from a detailed region", flush=True)

    report = {name: [] for name in sets}
    started = time.perf_counter()
    for kind, image, target, query_detail in items:
        # The site's rule: query + four corner pieces, a frame scores its best vector.
        q = embed(model, [image, *search.pieces(image)])
        q = torch.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
        sims = vectors @ q.T  # [F, 275, 5]
        for name, tiles in sets.items():
            best = sims.masked_fill(~tiles[:, :, None], -1.0).amax(dim=1)  # [F, 5]
            score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1))
            rank = int((score > score[target]).sum()) + 1
            report[name].append((kind, rank, abs(int(score.argmax()) - target) <= SAME_MOMENT,
                                 query_detail >= detailed_cut))
    print(f"scored in {time.perf_counter() - started:.0f}s", flush=True)

    names = list(sets)
    sizes = {name: float(sets[name].sum()) / count for name in names}
    for title, only_detailed in (("all queries", False), ("queries from a detailed region", True)):
        print(f"\n{args.film}, {count} frames, {title}. Cells: exact #1 / same moment (+-1 s) #1 / median rank\n")
        print("| type | n | " + " | ".join(f"{n} ({sizes[n]:.0f}/frame)" for n in names) + " |")
        print("|---|---|" + "---|" * len(names))
        for kind in [*KINDS, "ALL"]:
            cells, n = [], 0
            for name in names:
                sub = [r for r in report[name] if kind in ("ALL", r[0]) and (r[3] or not only_detailed)]
                n = len(sub)
                cells.append(f"{sum(r[1] == 1 for r in sub)} / {sum(r[2] for r in sub)} / "
                             f"{np.median([r[1] for r in sub]):.0f}" if sub else "-")
            print(f"| {kind} | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
