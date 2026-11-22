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

`--suite encodings` builds a second suite whose types are the real-world
encodings of curtain_ml.encodings instead (streams, rips, GIFs, image sites,
re-upload chains), each made from the target and its neighbouring frames.

`--suite stacked` goes through the steps a web screenshot does, in order:
maybe another release's grade, maybe a stream or rip, the screenshot itself
(the type: a crop, letterbox, subtitles, a phone photo), maybe an image
site. Featureless queries (black, flat) are drawn again. Its report adds
"same shot" and splits same-moment by stream and by site.

    .venv/bin/python -m rcmindex.validation_suite make --per-type 10
    .venv/bin/python -m rcmindex.validation_suite run
    .venv/bin/python -m rcmindex.validation_suite run --systems now
    .venv/bin/python -m rcmindex.validation_suite make --suite encodings --per-type 12
    .venv/bin/python -m rcmindex.validation_suite run --suite encodings
    .venv/bin/python -m rcmindex.validation_suite make --suite stacked --per-type 50
    .venv/bin/python -m rcmindex.validation_suite run --suite stacked --systems site
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from curtain_ml import encodings
from rcmindex import shots
from curtain_ml.model import CurtainEncoder
from curtain_ml.training import reference_transform
from rcmindex.data_paths import ORIGINAL_DIRS

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
SUITE = LOCAL / "benchmarks/validation-v1"
SUITES = {"validation": SUITE, "encodings": LOCAL / "benchmarks/encodings-v1",
          "stacked": LOCAL / "benchmarks/stacked-v1"}
NEIGHBOURS = 4  # frames either side of the target in an encodings clip
RESULTS = LOCAL / "benchmarks/results"
SOURCES = ORIGINAL_DIRS
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
    "regrade": lambda f, r: jpeg(_regrade(f, r), 85),
    "letterbox": lambda f, r: jpeg(_letterbox(f, r), 85),
    "subtitles": lambda f, r: jpeg(subtitle(crop(f, r, (0.4, 0.65), (0.4, 0.65)), r), 85),
    "phone_photo": lambda f, r: jpeg(_phone(f, r), 80),
}


def _regrade(frame: Image.Image, rng: random.Random) -> Image.Image:
    # Another release's colour grade: per-channel gain and gamma, crushed or
    # lifted blacks, a little saturation (like the Godfather screenshot: less
    # green and blue, deeper shadows).
    x = np.asarray(frame, dtype=np.float32) / 255
    gain = np.array([rng.uniform(0.85, 1.12) for _ in range(3)], dtype=np.float32)
    gamma = np.array([rng.uniform(0.9, 1.15) for _ in range(3)], dtype=np.float32)
    x = np.clip(x * gain, 1e-4, 1) ** gamma
    black = rng.uniform(-0.04, 0.05)
    x = np.clip((x - black) / (1 - black), 0, 1)
    grey = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    x = grey[..., None] + (x - grey[..., None]) * rng.uniform(0.8, 1.3)
    return Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8))


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


# The stacked suite, step by step. None: the step is skipped.
REGRADE_CHANCE = 0.4  # another release's grade, on every type but "regrade" (always)
STREAMS = ["stream_h264", "stream_hevc", "stream_av1_grain", "youtube_vp9", "dvd_mpeg2", None]
SITES = ["twitter", "reddit_preview", "whatsapp", None]
SCREENSHOTS = {
    "wide_crop": lambda f, r: crop(f, r, (0.7, 0.9), (0.7, 0.9)),
    "medium_crop": lambda f, r: crop(f, r, (0.4, 0.65), (0.4, 0.65)),
    "tight_crop": lambda f, r: crop(f, r, (0.2, 0.35), (0.2, 0.35)),
    "regrade": lambda f, r: f,
    "letterbox": lambda f, r: _player(f, r),
    "subtitles": lambda f, r: subtitle(crop(f, r, (0.4, 0.65), (0.4, 0.65)), r),
    "phone_photo": lambda f, r: _phone(f, r),
}


def _player(frame: Image.Image, rng: random.Random) -> Image.Image:
    # The frame fitted into a screen of another shape, bars where it falls
    # short (_letterbox adds no bars to a scope frame half the time).
    screen = rng.choice([16 / 9, 16 / 10, 4 / 3])
    w, h = frame.size
    if w / h > screen:
        canvas = Image.new("RGB", (w, round(w / screen)))
    else:
        canvas = Image.new("RGB", (round(h * screen), h))
    if canvas.size == frame.size:  # same shape: a 16:9 player around a 21:9 frame instead
        canvas = Image.new("RGB", (w, round(w * 9 / 16)))
    canvas.paste(frame, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def featureless(image: Image.Image) -> bool:
    # Black or flat: no system could place it (a black frame, a crop of night sky).
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return gray.std() < 12 or np.percentile(gray, 95) < 40


def stacked(kind: str, clip: list[Image.Image], rng: random.Random) -> tuple[Image.Image, dict]:
    steps = {"regrade": kind == "regrade" or rng.random() < REGRADE_CHANCE,
             "stream": rng.choice(STREAMS), "site": rng.choice(SITES)}
    if steps["regrade"]:  # the same grade on every frame of the clip
        grade = rng.random()
        clip = [_regrade(f, random.Random(grade)) for f in clip]
    frame = encodings.encode(steps["stream"], clip, NEIGHBOURS, rng) if steps["stream"] else clip[NEIGHBOURS]
    query = SCREENSHOTS[kind](frame, rng)
    if featureless(query):
        return None, steps
    if steps["site"]:
        query = encodings.encode(steps["site"], [query], 0, rng)
    return query, steps


def make(per_type: int, seed: int, suite: str) -> None:
    rng = random.Random(seed)
    folder = SUITES[suite]
    folder.mkdir(parents=True, exist_ok=True)
    frames = {m: sorted((SOURCES["full-curtain"] / m).glob("*.jpg")) for m in VALIDATION}
    types = {"encodings": encodings.RECIPES, "stacked": SCREENSHOTS}.get(suite, QUERY_TYPES)
    manifest = []
    for kind, build in types.items():
        (folder / kind).mkdir(exist_ok=True)
        for n in range(per_type):
            movie = VALIDATION[n % len(VALIDATION)]  # spread evenly over the movies
            steps = None
            if suite == "stacked":
                query = None
                while query is None:
                    at = rng.randrange(NEIGHBOURS, len(frames[movie]) - NEIGHBOURS)
                    source = frames[movie][at]
                    clip = [Image.open(p).convert("RGB") for p in frames[movie][at - NEIGHBOURS:at + NEIGHBOURS + 1]]
                    query, steps = stacked(kind, clip, rng)
                path = folder / kind / f"{n:02d}.png"
                query.save(path)
            elif suite == "encodings":  # keep the codec's output exactly: PNG
                at = rng.randrange(NEIGHBOURS, len(frames[movie]) - NEIGHBOURS)
                source = frames[movie][at]
                clip = [Image.open(p).convert("RGB") for p in frames[movie][at - NEIGHBOURS:at + NEIGHBOURS + 1]]
                query, path = build(clip, NEIGHBOURS, rng), folder / kind / f"{n:02d}.png"
                query.save(path)
            else:
                source = rng.choice(frames[movie])
                with Image.open(source) as image:
                    query = build(image.convert("RGB"), rng)
                path = folder / kind / f"{n:02d}.jpg"
                query.save(path, quality=95)
            manifest.append({"type": kind, "image": str(path.relative_to(folder)),
                             "target": f"full-curtain/{movie}/{source.name}", **({"steps": steps} if steps else {})})
    (folder / "manifest.json").write_text(json.dumps({"seed": seed, "per_type": per_type,
                                                      "movies": VALIDATION, "queries": manifest}, indent=1))
    print(f"wrote {len(manifest)} queries ({per_type} x {len(types)} types) to {folder}")


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


def system_now(pieces_on: bool = True, upload: bool = True, align_top: int = 0) -> callable:
    """The site now: combined model, FAISS over 4x4 tiles + whole frame,
    the query plus 4 pieces, exact re-scoring, 640 px browser upload; with
    align_top, the best align_top frames are re-ordered by SIFT alignment."""
    from concurrent.futures import ThreadPoolExecutor
    from rcmindex import align, search
    # The model must be the one whose arrays built the index (CURTAIN_FAISS_DIR).
    model = CurtainEncoder.from_checkpoint(Path(os.environ.get(
        "CURTAIN_CHECKPOINT", LOCAL / "trained_models/combined-rocm-50k-v1/best.pt")))
    engine = search.load()
    pool = ThreadPoolExecutor(8)

    def run(image: Image.Image) -> list[str] | tuple[list[str], dict]:
        original_width = image.width
        if upload:
            image = browser_upload(image)
        views = [image, *search.pieces(image)] if pieces_on else [image]
        q = embed(model, views)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        names = [engine["names"][f] for f, _ in search.search(engine, q, top=max(TOP, align_top))]
        if not align_top:
            return names[:TOP]
        query = align.features(np.asarray(image.convert("L")))

        def check(name: str):
            collection, rest = name.split("/", 1)
            return align.align(query, align.frame_features(SOURCES[collection] / rest))
        alignments = list(pool.map(check, names[:align_top]))
        order = align.rerank(names[:align_top], alignments)
        points, matrix = alignments[order[0]]
        if matrix:  # express it for the original query image (before the 640 px upload)
            s = image.width / original_width
            matrix = [v * s for v in matrix[:4]] + matrix[4:]
        return [names[i] for i in order][:TOP], {"points": points, "matrix": matrix}
    return run


def system_site(shots: bool = False) -> callable:
    """Exactly what the site runs (apps/server.py): every reading of the query,
    FAISS, alignment, refinement within the shot; with shots, one result per
    shot as the site's "group by shot" view shows them. Model and index as the
    server picks them (CURTAIN_CHECKPOINT, CURTAIN_FAISS_DIR)."""
    from apps.server import DEFAULT_CHECKPOINT, TrainedSearchEngine
    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT)
    names = engine.search_index["names"]

    def run(image: Image.Image) -> list[str]:
        image = browser_upload(image)
        ranked, _ = engine.rank(image, engine.embed_query(image), shots)
        return [names[entry[0]] for entry in ranked[:TOP]]
    return run


SYSTEMS = {
    "before": ("before: crop-v1, whole frames", system_before),
    "now-nopieces": ("now without pieces", lambda: system_now(pieces_on=False)),
    "now": ("now: tiles + pieces", system_now),
    "now-align5": ("now + alignment, top 5", lambda: system_now(align_top=5)),
    "now-align10": ("now + alignment, top 10", lambda: system_now(align_top=10)),
    "site": ("site", system_site),
    "site-shots": ("site, by shot", lambda: system_site(shots=True)),
}
DEFAULT_SYSTEMS = ["before", "now", "now-align10"]


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


def shot_maps() -> dict[str, np.ndarray]:
    # Shot number of each validation movie's frames (full-curtain copy), where mapped.
    out = {}
    for movie in VALIDATION:
        count = len(list((SOURCES["full-curtain"] / movie).glob("*.jpg")))
        if (shot := shots.load(f"full-curtain/{movie}", count)) is not None:
            out[f"full-curtain/{movie}"] = shot
    return out


def score(found: list[str], target: str, times: dict[str, float],
          maps: dict[str, np.ndarray] | None = None) -> dict[str, bool]:
    title = target.split("/")[1]

    def same_shot(name: str) -> bool:  # same full-curtain shot, else the same moment
        film = target.rsplit("/", 1)[0]
        if maps and film in maps and name.rsplit("/", 1)[0] == film:
            number = lambda n: int(n.rsplit("_", 1)[1].split(".")[0]) - 1
            return bool(maps[film][number(name)] == maps[film][number(target)])
        return same_moment(name)

    def same_moment(name: str) -> bool:  # either copy of a movie, within a second
        return (name.split("/")[1] == title and name in times and target in times
                and abs(times[name] - times[target]) <= SAME_MOMENT_SECONDS)
    return {
        "exact": bool(found) and found[0] == target,
        "top5": target in found,
        "moment": bool(found) and same_moment(found[0]),
        "movie": bool(found) and found[0].split("/")[1] == title,
        **({"shot": bool(found) and same_shot(found[0])} if maps is not None else {}),
    }


def run(selected: list[str], suite: str) -> None:
    folder = SUITES[suite]
    manifest = json.loads((folder / "manifest.json").read_text())
    queries = manifest["queries"]
    images = [Image.open(folder / q["image"]).convert("RGB") for q in queries]
    times = timestamps()
    maps = shot_maps() if suite == "stacked" else None
    torch.set_num_threads(16)
    report = {"suite": str(folder), "queries": len(queries), "systems": {}}
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
            found, extra = found if isinstance(found, tuple) else (found, {})
            rows.append({"type": query["type"], "image": query["image"], "found": found,
                         **({"steps": query["steps"]} if "steps" in query else {}), **extra, **score(found, query["target"], times, maps)})
        report["systems"][key] = {"label": label, "rows": rows,
                                  "median_ms": float(np.median(latencies) * 1000)}
        del system

    types = list(dict.fromkeys(q["type"] for q in queries))
    metrics = [("exact", "exact frame"), ("moment", "same moment (±1 s)"), ("top5", "exact in top 5"),
               ("movie", "right movie")] + ([("shot", "same shot")] if maps is not None else [])
    lines = [f"# {suite.capitalize()} suite: {len(queries)} queries, {len(types)} types, "
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
    for step in ["regrade", "stream", "site"] if "steps" in queries[0] else []:
        values = list(dict.fromkeys(str(q["steps"][step]) for q in queries))
        lines += [f"## same moment by {step}", "",
                  f"| {step} | " + " | ".join(report["systems"][k]["label"] for k in selected) + " |",
                  "|" + "---|" * (len(selected) + 1)]
        for value in values:
            cells = []
            for k in selected:
                rows = [r for r in report["systems"][k]["rows"] if str(r["steps"][step]) == value]
                cells.append(f"{sum(r['moment'] for r in rows)}/{len(rows)}")
            lines.append(f"| {value} | " + " | ".join(cells) + " |")
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
    r.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=list(SYSTEMS))
    for p in (m, r):
        p.add_argument("--suite", default="validation", choices=list(SUITES))
    args = parser.parse_args()
    if args.command == "make":
        make(args.per_type, args.seed, args.suite)
    else:
        run(args.systems, args.suite)


if __name__ == "__main__":
    main()
