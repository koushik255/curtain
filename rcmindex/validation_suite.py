"""Validation benchmark suite: fixed query images, every search system.

Queries come only from the validation movies (no model trained on them), in
eleven types, so systems can be compared per type:

  full_frame   whole frame, light re-compression
  wide_crop    70-90% of the width and height
  medium_crop  40-65% (a typical screenshot)
  tight_crop   20-35% (a face or object)
  narrow_crop  tall sliver: 15-25% wide, 60-90% high
  low_res      whole frame shrunk to 160-240 px wide
  heavy_jpeg   whole frame at JPEG quality 10-20
  color_shift  brightness, contrast, saturation and tint changed
  letterbox    frame inside black bars, as in a player screenshot
  subtitles    medium crop with a subtitle bar burned in
  phone_photo  slightly rotated, blurred and noisy, like a photo of a screen

`make` writes the images and a manifest once; `run` evaluates systems on
exactly those images and saves a JSON + Markdown report.

    .venv/bin/python -m rcmindex.validation_suite make --per-type 10
    .venv/bin/python -m rcmindex.validation_suite run
    .venv/bin/python -m rcmindex.validation_suite run --systems now
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import reference_transform

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
SUITE = LOCAL / "benchmarks/validation-v1"
RESULTS = LOCAL / "benchmarks/results"
SOURCES = {"full-curtain": LOCAL / "screenshots", "l4-50k-lbfive": Path.home() / "lbfive/screenshots"}
OLD_INDEXES = {"full-curtain": LOCAL / "trained_indexes/l4-50k-crop-v1",
               "l4-50k-lbfive": LOCAL / "trained_indexes/l4-50k-lbfive-crop-v1"}
# Validation movies of both checkpoints (their split.json files agree).
VALIDATION = ["Do_The_Right_Thing_1989", "Gladiator_2000", "The_Apartment_1960",
              "The_Handmaiden_2016", "The_King_Of_Comedy_1982", "The_Matrix_1999"]
TOP = 5
SAME_MOMENT_SECONDS = 1.0


# ------------------------------------------------------------------ queries

def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def crop(image: Image.Image, rng: random.Random, w: tuple[float, float], h: tuple[float, float]) -> Image.Image:
    cw, ch = int(image.width * rng.uniform(*w)), int(image.height * rng.uniform(*h))
    x, y = rng.randint(0, image.width - cw), rng.randint(0, image.height - ch)
    return image.crop((x, y, x + cw, y + ch))


def subtitle(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    bar = max(18, image.height // 9)
    top = image.height - bar - image.height // 20
    draw.rectangle((image.width * 0.1, top, image.width * 0.9, top + bar), fill=(0, 0, 0))
    font = ImageFont.load_default(size=max(10, bar * 2 // 3))
    text = rng.choice(["I'll be right back.", "What did you just say?", "We don't have much time.",
                       "Look at me.", "That's not how this works."])
    draw.text((image.width * 0.5, top + bar / 2), text, fill=(255, 255, 255), font=font, anchor="mm")
    return image


QUERY_TYPES = {
    "full_frame": lambda f, r: jpeg(f, r.randint(70, 90)),
    "wide_crop": lambda f, r: jpeg(crop(f, r, (0.7, 0.9), (0.7, 0.9)), 85),
    "medium_crop": lambda f, r: jpeg(crop(f, r, (0.4, 0.65), (0.4, 0.65)), 85),
    "tight_crop": lambda f, r: jpeg(crop(f, r, (0.2, 0.35), (0.2, 0.35)), 85),
    "narrow_crop": lambda f, r: jpeg(crop(f, r, (0.15, 0.25), (0.6, 0.9)), 85),
    "low_res": lambda f, r: jpeg(f.resize((w := r.randint(160, 240), round(f.height * w / f.width)),
                                          Image.BILINEAR), 85),
    "heavy_jpeg": lambda f, r: jpeg(f, r.randint(10, 20)),
    "color_shift": lambda f, r: jpeg(Image.blend(
        ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(f).enhance(r.uniform(0.7, 1.3)))
                           .enhance(r.uniform(0.7, 1.3))).enhance(r.uniform(0.6, 1.4)),
        Image.new("RGB", f.size, r.choice([(255, 170, 90), (90, 150, 255)])), r.uniform(0.05, 0.15)), 85),
    "letterbox": lambda f, r: jpeg(_letterbox(f, r), 85),
    "subtitles": lambda f, r: jpeg(subtitle(crop(f, r, (0.4, 0.65), (0.4, 0.65)), r), 85),
    "phone_photo": lambda f, r: jpeg(_phone(f, r), 80),
}


def _letterbox(frame: Image.Image, rng: random.Random) -> Image.Image:
    # Pad to a wider or taller player window with black bars.
    if rng.random() < 0.5:
        w, h = frame.width, round(frame.width * rng.uniform(0.75, 1.0))   # bars top/bottom
    else:
        w, h = round(frame.height * rng.uniform(1.9, 2.4)), frame.height  # bars left/right
    w, h = max(w, frame.width), max(h, frame.height)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(frame, ((w - frame.width) // 2, (h - frame.height) // 2))
    return canvas


def _phone(frame: Image.Image, rng: random.Random) -> Image.Image:
    rotated = frame.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, expand=False, fillcolor=(20, 20, 20))
    blurred = rotated.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.6)))
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, (frame.height, frame.width, 3))
    return Image.fromarray(np.clip(np.asarray(blurred, dtype=np.float32) + noise, 0, 255).astype(np.uint8))


def make(per_type: int, seed: int) -> None:
    rng = random.Random(seed)
    SUITE.mkdir(parents=True, exist_ok=True)
    frames = {m: sorted((SOURCES["full-curtain"] / m).glob("*.jpg")) for m in VALIDATION}
    manifest = []
    for kind, build in QUERY_TYPES.items():
        (SUITE / kind).mkdir(exist_ok=True)
        for n in range(per_type):
            movie = VALIDATION[n % len(VALIDATION)]  # spread evenly over the movies
            source = rng.choice(frames[movie])
            with Image.open(source) as image:
                query = build(image.convert("RGB"), rng)
            path = SUITE / kind / f"{n:02d}.jpg"
            query.save(path, quality=95)
            manifest.append({"type": kind, "image": str(path.relative_to(SUITE)),
                             "target": f"full-curtain/{movie}/{source.name}"})
    (SUITE / "manifest.json").write_text(json.dumps({"seed": seed, "per_type": per_type,
                                                     "movies": VALIDATION, "queries": manifest}, indent=1))
    print(f"wrote {len(manifest)} queries ({per_type} x {len(QUERY_TYPES)} types) to {SUITE}")


# ------------------------------------------------------------------ systems

def embed(model: CurtainEncoder, images: list[Image.Image]) -> np.ndarray:
    transform = reference_transform(model.image_size)
    with torch.inference_mode():
        return model(torch.stack([transform(i) for i in images])).float().numpy()


def browser_upload(image: Image.Image) -> Image.Image:
    # What the site's page uploads: 640 px on the long side, JPEG 90.
    scale = 640 / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    return jpeg(image, 90)


def system_before() -> callable:
    """The site before tiles: crop-v1 model, one whole-frame vector per frame."""
    model = CurtainEncoder.from_checkpoint(LOCAL / "trained_models/l4-50k-crop-v1-20260909/best.pt")
    names, blocks = [], []
    for collection, root in OLD_INDEXES.items():
        for movie_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            records = json.loads((movie_dir / "records.json").read_text())
            names += [f"{collection}/{r['movie']}/{r['filename']}" for r in records]
            blocks.append(np.load(movie_dir / "embeddings.npy"))
    matrix = torch.from_numpy(np.concatenate(blocks))  # fp16 [2.48M, 128]

    def run(image: Image.Image) -> list[str]:
        q = torch.from_numpy(embed(model, [image])[0]).half()
        scores = torch.cat([(matrix[i:i + (1 << 20)].float() @ q.float()) for i in range(0, len(matrix), 1 << 20)])
        return [names[i] for i in scores.topk(TOP).indices.tolist()]
    return run


def system_now(pieces_on: bool = True, upload: bool = True) -> callable:
    """The site now: combined model, FAISS over 4x4 tiles + whole frame,
    the query plus 4 pieces, exact re-scoring, 640 px browser upload."""
    from rcmindex import search
    model = CurtainEncoder.from_checkpoint(LOCAL / "trained_models/combined-rocm-50k-v1/best.pt")
    engine = search.load()

    def run(image: Image.Image) -> list[str]:
        if upload:
            image = browser_upload(image)
        views = [image, *search.pieces(image)] if pieces_on else [image]
        q = embed(model, views)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return [engine["names"][f] for f, _ in search.search(engine, q, top=TOP)]
    return run


SYSTEMS = {
    "before": ("before: crop-v1, whole frames", system_before),
    "now-nopieces": ("now without pieces", lambda: system_now(pieces_on=False)),
    "now": ("now: tiles + pieces (site)", system_now),
}


# ------------------------------------------------------------------ scoring

def timestamps() -> dict[str, float]:
    out = {}
    for collection, root in OLD_INDEXES.items():
        for movie in VALIDATION:
            path = root / movie / "records.json"
            if path.exists():
                for r in json.loads(path.read_text()):
                    out[f"{collection}/{movie}/{r['filename']}"] = r["timestamp_seconds"]
    return out


def score(found: list[str], target: str, times: dict[str, float]) -> dict[str, bool]:
    title = target.split("/")[1]

    def same_moment(name: str) -> bool:  # either copy of a movie, within a second
        return (name.split("/")[1] == title and name in times and target in times
                and abs(times[name] - times[target]) <= SAME_MOMENT_SECONDS)
    return {
        "exact": bool(found) and found[0] == target,
        "top5": target in found,
        "moment": bool(found) and same_moment(found[0]),
        "movie": bool(found) and found[0].split("/")[1] == title,
    }


def run(selected: list[str]) -> None:
    manifest = json.loads((SUITE / "manifest.json").read_text())
    queries = manifest["queries"]
    images = [Image.open(SUITE / q["image"]).convert("RGB") for q in queries]
    times = timestamps()
    torch.set_num_threads(16)
    report = {"suite": str(SUITE), "queries": len(queries), "systems": {}}
    for key in selected:
        label, build = SYSTEMS[key]
        started = time.perf_counter()
        system = build()
        print(f"{label}: loaded in {time.perf_counter() - started:.0f}s", flush=True)
        rows, latencies = [], []
        for query, image in zip(queries, images):
            t = time.perf_counter()
            found = system(image)
            latencies.append(time.perf_counter() - t)
            rows.append({"type": query["type"], "image": query["image"], "found": found,
                         **score(found, query["target"], times)})
        report["systems"][key] = {"label": label, "rows": rows,
                                  "median_ms": float(np.median(latencies) * 1000)}
        del system

    types = list(dict.fromkeys(q["type"] for q in queries))
    metrics = [("exact", "exact frame"), ("moment", "same moment (±1 s)"), ("top5", "exact in top 5"),
               ("movie", "right movie")]
    lines = [f"# Validation suite: {len(queries)} queries, {len(types)} types, "
             f"movies: {', '.join(m.replace('_', ' ') for m in manifest['movies'])}", ""]
    for metric, title in metrics:
        header = "| type | " + " | ".join(report["systems"][k]["label"] for k in selected) + " |"
        lines += [f"## {title}", "", header, "|" + "---|" * (len(selected) + 1)]
        for kind in types + ["ALL"]:
            cells = []
            for k in selected:
                rows = [r for r in report["systems"][k]["rows"] if kind in ("ALL", r["type"])]
                cells.append(f"{sum(r[metric] for r in rows)}/{len(rows)}")
            lines.append(f"| {'**all**' if kind == 'ALL' else kind} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["median search time: " + ", ".join(f"{report['systems'][k]['label']} "
                                                  f"{report['systems'][k]['median_ms']:.0f} ms" for k in selected)]
    text = "\n".join(lines)
    print("\n" + text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(report, indent=1))
    (RESULTS / f"{stamp}.md").write_text(text + "\n")
    print(f"\nsaved {RESULTS / stamp}.json and .md")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--per-type", type=int, default=10)
    m.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run")
    r.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    args = parser.parse_args()
    if args.command == "make":
        make(args.per_type, args.seed)
    else:
        run(args.systems)


if __name__ == "__main__":
    main()
