"""Can the index be rebuilt from the 640 px WebP copies instead of the originals?

Only the validation movies still have their full-size (1280 px) originals;
every other movie is kept as 640 px WebP copies. rcmindex resizes every frame
to 768x768 before tiling, so a WebP copy loses about half its vertical detail
to the original. This indexes the six validation movies (still whole on disk)
from each source and runs the same queries against each:

  original   the 1280 px JPEGs
  webp640    640 px wide, WebP q75: exactly what .local/frames-small/ keeps
  webp960    960 px wide, WebP q75: the middle ground if 640 falls short

Embeddings are computed on the CPU with rcmindex's method (export_model.py
shared_trunk: one trunk pass at 768x768, 49 pooled tiles plus the whole
frame), for every source alike. Search is brute force over the six movies'
frames, scored like search.search (query plus 4 pieces, after the site's
640 px browser upload), then SIFT alignment re-ranks the top 10 against that
source's own images, as the site would. Only six movies are searched, so the
rates are higher than the README's; the comparison between sources is what
counts.

Each step is resumable: WebP copies and per-movie embeddings are kept under
.local/webp-test/ and skipped when present.

    .venv/bin/python -m rcmindex.benchmark_webp --extra 30
    .venv/bin/python -m rcmindex.benchmark_webp --frames 200 --extra 0   # quick smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from curtain_ml.model import CurtainEncoder
from rcmindex import align, search
from rcmindex.benchmark_int8 import ALIGN_TOP, queries
from rcmindex.export_model import GRID, STEPS, normalized, trunk_of
from rcmindex.validation_suite import LOCAL, SOURCES, VALIDATION, browser_upload, embed, score, timestamps

OUT = LOCAL / "webp-test"
MODEL = LOCAL / "trained_models/combined-rocm-50k-v1/best.pt"
VARIANTS = {"original": None, "webp640": 640, "webp960": 960}
QUALITY = 75  # what .local/frames-small/ was made with
BATCH = 16
started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def originals(movie: str, limit: int) -> list[Path]:
    frames = sorted((SOURCES["full-curtain"] / movie).glob("*.jpg"))
    return frames[:limit] if limit else frames


def source(variant: str, frame: Path) -> Path:
    if VARIANTS[variant] is None:
        return frame
    return OUT / variant / frame.parent.name / (frame.stem + ".webp")


def shrink(job: tuple[Path, Path, int]) -> None:
    # As .local/frames-small/ was made, from the original JPEG.
    frame, target, width = job
    with Image.open(frame) as image:
        image.draft("RGB", (width, width))
        image = image.convert("RGB")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=QUALITY, method=2)
    target.write_bytes(buffer.getvalue())


def make_copies(variant: str, frames: list[Path]) -> None:
    width = VARIANTS[variant]
    todo = [(f, source(variant, f), width) for f in frames if not source(variant, f).is_file()]
    if not todo:
        return
    for directory in {t.parent for _, t, _ in todo}:
        directory.mkdir(parents=True, exist_ok=True)
    say(f"{variant}: writing {len(todo):,} WebP copies")
    with ProcessPoolExecutor(16) as pool:
        list(pool.map(shrink, todo, chunksize=64))
    size = sum(source(variant, f).stat().st_size for f in frames)
    say(f"{variant}: {size / len(frames) / 1024:.1f} KB per frame on average")


def load(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1)


def frame_vectors(model: CurtainEncoder, trunk: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    """uint8 [B, 3, H, W] -> [B, 50, 128] fp16: 49 tiles then the whole frame,
    as export_model.shared_trunk computes them, batched."""
    features = trunk(normalized(images, GRID * model.image_size))
    cells = features.shape[-1] // GRID
    stride = cells // 2
    pooled = torch.stack([
        features[:, :, y * stride:y * stride + cells, x * stride:x * stride + cells].mean((2, 3))
        for y in range(STEPS) for x in range(STEPS)
    ], 1)
    tiles = torch.nn.functional.normalize(model.backbone.fc(pooled), dim=-1)
    whole = model(normalized(images, model.image_size))
    return torch.cat([tiles, whole[:, None]], 1).half().numpy()


def embed_movie(model: CurtainEncoder, trunk: torch.nn.Module, variant: str, movie: str,
                frames: list[Path]) -> np.ndarray:
    selection = hashlib.sha1("\n".join(f.name for f in frames).encode()).hexdigest()[:10]
    path = OUT / "vectors" / variant / f"{movie}-{len(frames)}-{selection}.npy"
    if path.is_file():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(frames), 50, 128), dtype=np.float16)
    reader = ThreadPoolExecutor(4)
    t = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(frames), BATCH):
            batch = list(reader.map(load, [source(variant, f) for f in frames[start:start + BATCH]]))
            out[start:start + len(batch)] = frame_vectors(model, trunk, torch.stack(batch))
            if start and start % (BATCH * 200) == 0:
                rate = start / (time.monotonic() - t)
                say(f"  {variant} {movie}: {start:,}/{len(frames):,} frames, {rate:.0f}/s")
    np.save(path, out)
    say(f"{variant} {movie}: {len(frames):,} frames in {time.monotonic() - t:.0f}s")
    return out


def ranked(vectors: torch.Tensor, q: np.ndarray) -> np.ndarray:
    # search.search's scoring rule (as benchmark_int8.rank), brute force over every frame.
    n = len(vectors)
    best = (vectors.reshape(n * 50, 128) @ torch.from_numpy(q).T).reshape(n, 50, -1).amax(1)
    score_ = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(1))
    return score_.topk(ALIGN_TOP).indices.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", type=int, default=30, help="fresh queries per type on top of the suite")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--frames", type=int, default=0, help="only the first N frames per movie (smoke test)")
    parser.add_argument("--every", type=int, default=1,
                        help="index every Nth frame, plus all frames within --window of a query's target")
    parser.add_argument("--window", type=int, default=120, help="frames either side of each target (120 = 1 min)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()
    torch.set_num_threads(16)

    frames = {m: originals(m, args.frames) for m in VALIDATION}
    items = queries(args.extra, args.seed)
    if args.every > 1:
        # Every frame near a query's target (the hard look-alikes for "exact"), plus
        # every Nth frame of the rest of the movie as distractors.
        for movie in VALIDATION:
            position = {f.name: i for i, f in enumerate(frames[movie])}
            targets = [position[t.rsplit("/", 1)[1]] for _, _, t in items
                       if t.split("/")[1] == movie and t.rsplit("/", 1)[1] in position]
            near = np.zeros(len(frames[movie]), dtype=bool)
            for t in targets:
                near[max(0, t - args.window):t + args.window + 1] = True
            frames[movie] = [f for i, f in enumerate(frames[movie]) if near[i] or i % args.every == 0]
    names = [f"full-curtain/{m}/{f.name}" for m in VALIDATION for f in frames[m]]
    paths = {variant: [source(variant, f) for m in VALIDATION for f in frames[m]] for variant in args.variants}
    say(f"{len(names):,} frames in {len(VALIDATION)} movies, variants {', '.join(args.variants)}")
    for variant in args.variants:
        if VARIANTS[variant] is not None:
            make_copies(variant, [f for m in VALIDATION for f in frames[m]])

    model = CurtainEncoder.from_checkpoint(MODEL).eval()
    trunk = trunk_of(model)
    vectors = {variant: np.concatenate([embed_movie(model, trunk, variant, m, frames[m]) for m in VALIDATION])
               for variant in args.variants}
    if "original" in vectors:
        for variant in vectors:
            if variant != "original":
                cos = (vectors[variant].astype(np.float32) * vectors["original"].astype(np.float32)).sum(-1)
                say(f"{variant} vs original, per vector cosine: mean {cos.mean():.4f}  "
                    f"p5 {np.percentile(cos, 5):.4f}  min {cos.min():.4f}")

    vectors = {variant: torch.from_numpy(v.astype(np.float32)) for variant, v in vectors.items()}
    known = set(names)
    items = [i for i in items if i[2] in known]
    say(f"{len(items)} queries")
    times = timestamps()
    pool = ThreadPoolExecutor(8)
    rows = []
    for number, (kind, image, target) in enumerate(items, 1):
        image = browser_upload(image)
        q = embed(model, [image, *search.pieces(image)])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        query = align.features(np.asarray(image.convert("L")))
        row = {"type": kind, "target": target}
        for variant, v in vectors.items():
            order = ranked(v, q)
            plain = [names[f] for f in order[:5]]
            top = order[:ALIGN_TOP]
            checked = list(pool.map(lambda f: align.align(query, align.frame_features(paths[variant][f])), top))
            aligned = [names[f] for f in top[align.rerank(top, checked)][:5]]
            row[variant] = {"plain": score(plain, target, times), "aligned": score(aligned, target, times),
                            "found": aligned[0]}
        rows.append(row)
        if number % 50 == 0:
            say(f"  {number}/{len(items)} queries")

    types = list(dict.fromkeys(r["type"] for r in rows))
    variants = list(vectors)
    lines = [f"# Re-indexing from WebP copies: {len(rows)} queries, {len(names):,} frames, "
             f"{len(VALIDATION)} validation movies", ""]
    for stage, title in (("aligned", "with SIFT alignment of the top 10 (the site)"), ("plain", "ranking only")):
        lines += [f"## {title}", "", "| metric | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
        for metric in ("exact", "moment", "top5", "movie"):
            lines.append(f"| {metric} | " + " | ".join(
                f"{sum(r[v][stage][metric] for r in rows)}/{len(rows)}" for v in variants) + " |")
        lines.append("")
    lines += ["## exact frame by type (with alignment)", "", "| type | " + " | ".join(variants) + " |",
              "|---|" + "---|" * len(variants)]
    for kind in types:
        sub = [r for r in rows if r["type"] == kind]
        lines.append(f"| {kind} | " + " | ".join(f"{sum(r[v]['aligned']['exact'] for r in sub)}/{len(sub)}"
                                              for v in variants) + " |")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = LOCAL / "benchmarks/results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"webp-{stamp}.json").write_text(json.dumps(rows, indent=1))
    (results / f"webp-{stamp}.md").write_text(text + "\n")
    say(f"saved {results / f'webp-{stamp}'}.json and .md")


if __name__ == "__main__":
    main()
