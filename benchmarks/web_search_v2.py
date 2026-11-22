"""Image-search screenshots against the live site, broken down by artifact.

online_benchmark.py (v1) mostly used clean film-grab galleries for 95 of the
first 474 films. This takes what an image search actually returns for a film
(reposts, news-site stills, wallpapers, recompressed thumbnails) for a fixed
sample of the 841 indexed films, most of them films indexed after v1:

    collect   Bing image search "<title> <year> movie scene" per film; keep up to
              --per-film landscape, non-duplicate images (posters are portrait).
    sheets    contact sheets to review before any search runs; write
              reviews.json {id: "frame" | "overlay" | "still" | "exclude"}
              (overlay: a frame under a watermark, logo or subtitle).
    measure   artifact measurements per image: JPEG quality, size, noise/grain,
              blockiness, sharpness, letterboxing.
    run       send each image and its stress variants (real video codecs via
              ffmpeg, synthetic film grain, heavy JPEG, downscaling) to the
              live /api/search-stream; resumable.
    report    accuracy per variant and per measured-artifact bucket.

Movie identification only; exact frames are not labelled. Image search
results include publicity stills that are not frames of the film; review
marks them "still" so they are reported separately.

    .venv/bin/python -m benchmarks.web_search_v2 collect
    .venv/bin/python -m benchmarks.web_search_v2 sheets
    .venv/bin/python -m benchmarks.web_search_v2 measure
    .venv/bin/python -m benchmarks.web_search_v2 run
    .venv/bin/python -m benchmarks.web_search_v2 report
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import random
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local/benchmarks/web-search-v2"
BASE_URL = "http://127.0.0.1:8781"
SEED = 20261121
BROWSER = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
# Films indexed before the v2 index; the rest were added 2026-11-20.
V1_CATALOG = ROOT / ".local/faiss-all-films-v1/catalog.json"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 20, data: bytes | None = None) -> bytes:
    headers = {"User-Agent": BROWSER}
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def title_of(movie: dict) -> str:
    title = re.sub(r"\b(\w+) s\b", r"\1's", movie["title"])  # "A Dog s Will" -> "A Dog's Will"
    return f"{title} {movie['year']}"


# ---------------------------------------------------------------- collect

def bing_images(query: str) -> list[dict]:
    url = "https://www.bing.com/images/search?" + urllib.parse.urlencode({"q": query, "form": "HDRSC2", "first": 1})
    page = fetch(url).decode("utf-8", errors="replace")
    found = []
    for blob in re.findall(r'class="iusc"[^>]*\sm="([^"]+)"', page):
        meta = json.loads(html.unescape(blob))
        if meta.get("murl"):
            found.append({"image_url": meta["murl"], "page_url": meta.get("purl"), "title": meta.get("t", "")})
    return found


def dhash(image: Image.Image) -> int:
    thumb = np.asarray(image.convert("L").resize((9, 8)))
    return int("".join("1" if b else "0" for b in (thumb[:, 1:] > thumb[:, :-1]).ravel()), 2)


def collect_film(movie: dict, out: Path, per_film: int) -> dict:
    query = f"{title_of(movie)} movie scene"
    ledger = {"movie": movie["id"], "query": query, "cohort": movie["cohort"], "attempts": [], "queries": []}
    try:
        ledger["results"] = bing_images(query)
    except OSError as exc:
        ledger["error"] = str(exc)
        return ledger
    hashes = []
    for rank, hit in enumerate(ledger["results"]):
        if len(ledger["queries"]) >= per_film:
            break
        attempt = {**hit, "rank": rank}
        try:
            data = fetch(hit["image_url"])
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                fmt = im.format
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            w, h = rgb.size
            if min(w, h) < 120:
                raise ValueError(f"too small {w}x{h}")
            if w / h < 1.2:
                raise ValueError(f"not landscape {w}x{h} (poster or portrait)")
            d = dhash(rgb)
            if any((d ^ old).bit_count() <= 4 for old in hashes):
                raise ValueError("near-duplicate of a kept image")
            hashes.append(d)
            key = f"{movie['id']}__{len(ledger['queries']):02d}"
            (out / "originals" / key).write_bytes(data)
            ledger["queries"].append({"id": key, "movie": movie["id"], "cohort": movie["cohort"], "rank": rank,
                                      "image_url": hit["image_url"], "page_url": hit["page_url"],
                                      "title": hit["title"], "format": fmt, "width": w, "height": h,
                                      "bytes": len(data), "sha256": sha(data)})
            attempt["selected"] = key
        except Exception as exc:  # any broken host or undecodable file: next result
            attempt["error"] = f"{type(exc).__name__}: {exc}"[:200]
        ledger["attempts"].append(attempt)
    return ledger


def collect(args):
    movies = read_json_url(args.base_url + "/api/movies")["movies"]
    old = {m["movie"].split("/")[1] for m in read(V1_CATALOG)["movies"]}
    for m in movies:
        m["cohort"] = "v1" if m["id"] in old else "new"
    rng = random.Random(SEED)
    new = sorted((m for m in movies if m["cohort"] == "new"), key=lambda m: m["id"].casefold())
    v1 = sorted((m for m in movies if m["cohort"] == "v1"), key=lambda m: m["id"].casefold())
    sample = rng.sample(new, min(args.new_films, len(new))) + rng.sample(v1, min(args.v1_films, len(v1)))
    if args.ids:
        sample = [m for m in movies if m["id"] in args.ids]
    (args.out / "originals").mkdir(parents=True, exist_ok=True)
    ledgers = args.out / "acquisition"
    ledgers.mkdir(exist_ok=True)
    todo = [m for m in sample if not (ledgers / f"{m['id']}.json").exists()]
    with ThreadPoolExecutor(args.workers) as pool:
        for ledger in pool.map(lambda m: collect_film(m, args.out, args.per_film), todo):
            save(ledgers / f"{ledger['movie']}.json", ledger)
            print(f"{ledger['movie']}: {len(ledger['queries'])} images {ledger.get('error', '')}", flush=True)
    queries = []
    for m in sample:
        path = ledgers / f"{m['id']}.json"
        if path.exists():
            queries += read(path)["queries"]
    save(args.out / "manifest.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "seed": SEED, "films": [m["id"] for m in sample], "queries": queries})
    print(f"{len(queries)} images for {len({q['movie'] for q in queries})}/{len(sample)} films", flush=True)


def read_json_url(url: str):
    return json.loads(fetch(url, timeout=60))


# ---------------------------------------------------------------- review sheets

def sheets(args):
    queries = read(args.out / "manifest.json")["queries"]
    directory = args.out / "sheets"
    directory.mkdir(exist_ok=True)
    per = 20
    for start in range(0, len(queries), per):
        group = queries[start:start + per]
        canvas = Image.new("RGB", (1600, 250 * ((len(group) + 3) // 4)), "#202020")
        draw = ImageDraw.Draw(canvas)
        for i, q in enumerate(group):
            x, y = i % 4 * 400, i // 4 * 250
            with Image.open(args.out / "originals" / q["id"]) as im:
                canvas.paste(ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (392, 218)), (x + 4, y + 26))
            draw.text((x + 4, y + 6), f"{start + i} {q['id'][:52]}", fill="white")
        canvas.save(directory / f"{start // per:03d}.jpg", quality=85)
    print(f"{(len(queries) + per - 1) // per} sheets in {directory}", flush=True)


# ---------------------------------------------------------------- measurements

STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def jpeg_quality(im: Image.Image) -> float | None:
    """IJG quality that best reproduces the luma table (None if not JPEG)."""
    tables = getattr(im, "quantization", None)
    if not tables or 0 not in tables:
        return None
    table = np.array(tables[0], dtype=np.float64)
    best, err = None, 1e18
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        guess = np.clip(np.floor((STD_LUMA * scale + 50) / 100), 1, 255)
        e = np.abs(np.sort(guess) - np.sort(table)).sum()  # sorted: zigzag vs natural order
        if e < err:
            best, err = q, e
    return float(best)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate (grain plus sensor/codec noise), 8-bit units."""
    g = gray.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    conv = sum(k[i, j] * g[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3))
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def blockiness(gray: np.ndarray) -> float:
    """Mean column/row step across 8-px block edges over the step inside blocks; ~1 clean, >1.3 blocky."""
    g = gray.astype(np.float64)
    dx, dy = np.abs(np.diff(g, axis=1)).mean(axis=0), np.abs(np.diff(g, axis=0)).mean(axis=1)
    def ratio(d):
        edge = d[7::8]
        inner = np.delete(d, np.arange(7, len(d), 8))
        return edge.mean() / max(inner.mean(), 1e-6)
    return float((ratio(dx) + ratio(dy)) / 2)


def measure_one(path: Path) -> dict:
    with Image.open(path) as im:
        im.load()
        q = jpeg_quality(im)
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float64)
    rows = gray.mean(axis=1)
    dark = rows < 18
    bars = (np.argmax(~dark) if (~dark).any() else 0) + (np.argmax(~dark[::-1]) if (~dark).any() else 0)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    # Noise on a fixed scale so a 4K wallpaper and a 300 px thumbnail compare.
    norm = np.asarray(ImageOps.contain(rgb, (960, 960)).convert("L"))
    return {"jpeg_quality": q, "long_side": max(rgb.size), "noise": noise_sigma(norm),
            "blockiness": blockiness(gray), "sharpness": float(lap.var()),
            "letterbox": float(bars / gray.shape[0])}


def measure(args):
    queries = read(args.out / "manifest.json")["queries"]
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(lambda q: measure_one(args.out / "originals" / q["id"]), queries))
    save(args.out / "measurements.json", {q["id"]: v for q, v in zip(queries, values)})
    print(f"measured {len(values)} images", flush=True)


# ---------------------------------------------------------------- stress variants

def even_rgb(im: Image.Image, long_side: int = 960) -> Image.Image:
    im = ImageOps.contain(im, (long_side, long_side)) if max(im.size) > long_side else im
    return im.crop((0, 0, im.width // 2 * 2, im.height // 2 * 2))


def video_codec(im: Image.Image, args_: list[str], ext: str) -> Image.Image:
    """Encode one frame with a real video encoder (yuv420p) and decode it back."""
    im = even_rgb(im)
    with tempfile.TemporaryDirectory() as tmp:
        src, enc, dec = Path(tmp, "in.png"), Path(tmp, "enc" + ext), Path(tmp, "out.png")
        im.save(src)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-pix_fmt", "yuv420p", *args_, "-frames:v", "1", enc],
                       stderr=subprocess.DEVNULL,
                       check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", enc, "-frames:v", "1", dec], check=True)
        with Image.open(dec) as out:
            return out.convert("RGB")


def film_grain(im: Image.Image, sigma: float, seed: int) -> Image.Image:
    """Luma grain, spatially correlated (generated at half resolution) and
    strongest in midtones, as film grain is."""
    a = np.asarray(im, dtype=np.float32)
    rng = np.random.default_rng(seed)
    h, w = a.shape[:2]
    noise = rng.standard_normal(((h + 1) // 2, (w + 1) // 2)).astype(np.float32)
    noise = np.array(Image.fromarray(noise, mode="F").resize((w, h), Image.Resampling.BILINEAR))
    noise /= noise.std() + 1e-6
    luma = a @ np.array([.299, .587, .114], dtype=np.float32) / 255
    weight = 0.4 + 2.4 * luma * (1 - luma)  # 0.4 in shadows/highlights, 1.0 at mid grey
    return Image.fromarray(np.clip(a + (sigma * weight * noise)[..., None], 0, 255).astype(np.uint8))


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def variant_image(im: Image.Image, name: str, seed: int) -> Image.Image:
    if name == "grain_light":
        return film_grain(even_rgb(im), 6, seed)
    if name == "grain_heavy":
        return film_grain(even_rgb(im), 14, seed)
    if name == "h264_crf40":
        return video_codec(im, ["-c:v", "libx264", "-crf", "40", "-preset", "medium"], ".mp4")
    if name == "hevc_crf40":
        return video_codec(im, ["-c:v", "libx265", "-crf", "40", "-x265-params", "log-level=error"], ".mp4")
    if name == "av1_crf55":
        return video_codec(im, ["-c:v", "libsvtav1", "-crf", "55", "-preset", "8"], ".mp4")
    if name == "vp9_crf55":
        return video_codec(im, ["-c:v", "libvpx-vp9", "-crf", "55", "-b:v", "0"], ".webm")
    if name == "grain_h264":  # grain is what low-bitrate encoders smear into blocks
        return video_codec(film_grain(even_rgb(im), 14, seed), ["-c:v", "libx264", "-crf", "36"], ".mp4")
    if name == "jpeg15":
        return jpeg(even_rgb(im), 15)
    if name == "small240":
        return ImageOps.contain(im, (240, 240))
    raise ValueError(name)


VARIANTS = ["original", "grain_light", "grain_heavy", "h264_crf40", "hevc_crf40", "av1_crf55", "vp9_crf55",
            "grain_h264", "jpeg15", "small240"]


def payload(path: Path, name: str, seed: int) -> bytes:
    """The variant, then what the site's upload page does to every image
    (apps/templates/index.html): longest side at most 640 px, JPEG quality 90."""
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
    if name != "original":
        im = variant_image(im, name, seed)
    if max(im.size) > 640:
        im = ImageOps.contain(im, (640, 640), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------- run / report

def search(base_url: str, data: bytes) -> dict:
    events = [json.loads(line) for line in fetch(base_url + "/api/search-stream", timeout=180, data=data).splitlines()
              if line.strip()]
    errors = [e for e in events if e.get("type") == "error"]
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if errors or result is None:
        raise ValueError(str(errors or "no result event"))
    return result


def run(args):
    queries = read(args.out / "manifest.json")["queries"]
    reviews = read(args.out / "reviews.json") if (args.out / "reviews.json").exists() else {}
    queries = [q for q in queries if reviews.get(q["id"], "frame") != "exclude"]
    if args.ids:
        queries = [q for q in queries if q["id"] in args.ids or q["movie"] in args.ids]
    variants = args.variants or VARIANTS
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {(r["id"], r["variant"]) for r in map(json.loads, results_path.read_text().splitlines())}
    health = read_json_url(args.base_url + "/api/health")
    save(args.out / "run-health.json", health)
    jobs = [(q, v) for v in variants for q in queries if (q["id"], v) not in done]
    print(f"{len(jobs)} searches ({len(queries)} images x {len(variants)} variants, {len(done)} done); {health}",
          flush=True)

    def one(job):
        q, v = job
        row = {"id": q["id"], "movie": q["movie"], "cohort": q["cohort"], "variant": v}
        try:
            data = payload(args.out / "originals" / q["id"], v, int(sha(q["id"].encode())[:8], 16))
            started = time.perf_counter()
            result = search(args.base_url, data)
            found = result.get("results", [])
            top = found[0] if found else {}
            row.update(seconds=time.perf_counter() - started, payload_bytes=len(data),
                       top1=top.get("movie") == q["movie"],
                       top5=any(r.get("movie") == q["movie"] for r in found[:5]),
                       top20=any(r.get("movie") == q["movie"] for r in found),
                       confirmed=top.get("movie") == q["movie"] and bool(top.get("confirmed")),
                       weak=result.get("weak"), points=top.get("points", 0),
                       got=f"{top.get('movie')}/{top.get('filename')}",
                       results=[{k: r.get(k) for k in ("movie", "filename", "source", "points", "confirmed")}
                                for r in found[:20]])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return row

    with results_path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(one, jobs), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            status = row.get("error") or ("ok" if row["top1"] else f"WRONG -> {row['got']}")
            print(f"{n}/{len(jobs)} {row['id']} {row['variant']}: {status}", flush=True)


def pct(rows, key):
    return f"{sum(r[key] for r in rows)}/{len(rows)} ({100 * sum(r[key] for r in rows) / max(len(rows), 1):.0f}%)"


def report(args):
    rows = [r for r in map(json.loads, (args.out / "results.jsonl").read_text().splitlines()) if "error" not in r]
    errors = sum(1 for line in (args.out / "results.jsonl").read_text().splitlines() if '"error"' in line)
    reviews = read(args.out / "reviews.json") if (args.out / "reviews.json").exists() else {}
    meas = read(args.out / "measurements.json")
    health = read(args.out / "run-health.json")
    lines = [f"# Image-search screenshots, {time.strftime('%Y-%m-%d')}", "",
             f"Live site: {health.get('movies')} films, {health.get('frames'):,} frames. "
             f"{len({r['id'] for r in rows})} images, {len({r['movie'] for r in rows})} films. "
             "Movie identification only (exact frames unlabelled).", ""]
    if errors:
        lines += [f"{errors} searches errored and are left out.", ""]

    def table(title, groups):
        nonlocal lines
        lines += [f"## {title}", "", "| | Images | Right film #1 | Right film top 5 | In top 20 | #1 confirmed |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, g in groups:
            if g:
                lines.append(f"| {name} | {len(g)} | {pct(g, 'top1')} | {pct(g, 'top5')} | {pct(g, 'top20')} | "
                             f"{pct(g, 'confirmed')} |")
        lines.append("")

    kind = lambda r: reviews.get(r["id"], "frame")
    frames = [r for r in rows if kind(r) in ("frame", "overlay")]
    table("Every variant (film frames, with or without overlays)",
          [(v, [r for r in frames if r["variant"] == v]) for v in VARIANTS])
    original = [r for r in rows if r["variant"] == "original"]
    table("Original images by source type", [
        ("clean film frames", [r for r in original if kind(r) == "frame"]),
        ("frames under a watermark/logo/subtitle", [r for r in original if kind(r) == "overlay"]),
        ("publicity stills (may not be in the footage)", [r for r in original if kind(r) == "still"]),
        ("frames, films indexed before v2", [r for r in original if r["cohort"] == "v1" and kind(r) in ("frame", "overlay")]),
        ("frames, films added in v2", [r for r in original if r["cohort"] == "new" and kind(r) in ("frame", "overlay")])])
    of = [r for r in original if kind(r) == "frame"]
    m = lambda r: meas[r["id"]]
    noise = sorted(m(r)["noise"] for r in of)
    q1, q3 = (noise[len(noise) // 4], noise[3 * len(noise) // 4]) if noise else (0, 0)
    table("Original clean film frames by measured artifact", [
        ("JPEG quality >= 85", [r for r in of if (m(r)["jpeg_quality"] or 100) >= 85]),
        ("JPEG quality 60-84", [r for r in of if m(r)["jpeg_quality"] and 60 <= m(r)["jpeg_quality"] < 85]),
        ("JPEG quality < 60", [r for r in of if m(r)["jpeg_quality"] and m(r)["jpeg_quality"] < 60]),
        ("not JPEG (WebP/PNG/...)", [r for r in of if m(r)["jpeg_quality"] is None]),
        ("blocky (blockiness > 1.3)", [r for r in of if m(r)["blockiness"] > 1.3]),
        (f"least noise/grain (sigma < {q1:.1f})", [r for r in of if m(r)["noise"] < q1]),
        (f"most noise/grain (sigma > {q3:.1f})", [r for r in of if m(r)["noise"] > q3]),
        ("long side < 500 px", [r for r in of if m(r)["long_side"] < 500]),
        ("long side 500-1199 px", [r for r in of if 500 <= m(r)["long_side"] < 1200]),
        ("long side >= 1200 px", [r for r in of if m(r)["long_side"] >= 1200]),
        ("letterboxed (bars > 5%)", [r for r in of if m(r)["letterbox"] > 0.05])])
    lost = {}
    by = {(r["id"], r["variant"]): r for r in frames}
    for (i, v), r in by.items():
        base = by.get((i, "original"))
        if v != "original" and base and base["top1"] and not r["top1"]:
            lost.setdefault(v, []).append(i)
    lines += ["## Found as original, lost after the variant", ""]
    lines += [f"- {v}: {len(ids)}" for v, ids in sorted(lost.items(), key=lambda kv: -len(kv[1]))] + [""]
    wrong = [r for r in of if not r["top1"]]
    lines += ["## Wrong film on the original image (clean frames)", ""]
    lines += [f"- {r['id']} -> {r['got']} ({r['points']} points)" for r in wrong] + [""]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------- misses page

FRAME_ROOTS = (ROOT / ".local/data/frames/originals", ROOT / ".local/data/frames/webp")


def frame_file(result: dict) -> Path | None:
    stem = Path(result["filename"]).stem
    for path in (FRAME_ROOTS[0] / result["source"] / result["filename"], FRAME_ROOTS[1] / result["source"] / f"{stem}.webp"):
        if path.is_file():
            return path
    return None


def misses(args):
    """Static page of every original image whose first result was the wrong film:
    the web image, what the site returned, and the right film's best frame if it
    was anywhere in the 20 results."""
    reviews = read(args.out / "reviews.json")
    manifest = {q["id"]: q for q in read(args.out / "manifest.json")["queries"]}
    rows = [r for r in map(json.loads, (args.out / "results.jsonl").read_text().splitlines())
            if r["variant"] == "original" and "error" not in r and not r["top1"]]
    site = args.out / "misses-site"
    images = site / "img"
    images.mkdir(parents=True, exist_ok=True)

    def jpg(source: Path, name: str) -> str:
        with Image.open(source) as im:
            im = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (640, 640))
        im.save(images / name, quality=85)
        return f"img/{name}"

    names = {"frame": "Clean film frame", "overlay": "Frame with watermark / logo / subtitle", "still": "Publicity still"}
    order = {"frame": 0, "overlay": 1, "still": 2}
    rows.sort(key=lambda r: (order[reviews[r["id"]]], r["movie"].casefold(), r["id"]))
    cards = []
    for r in rows:
        q = manifest[r["id"]]
        got = r["results"][0] if r["results"] else None
        right = next(((i, x) for i, x in enumerate(r["results"]) if x["movie"] == r["movie"]), None)
        panels = [("Web image", jpg(args.out / "originals" / r["id"], f"{r['id']}.q.jpg"), "")]
        if got and (f := frame_file(got)):
            panels.append((f"Site's #1: {got['movie'].replace('_', ' ')}", jpg(f, f"{r['id']}.got.jpg"),
                           f"{got['filename']} &middot; {got['points']} points{' &middot; confirmed' if got['confirmed'] else ''}"))
        if right and (f := frame_file(right[1])):
            panels.append((f"Right film, ranked #{right[0] + 1}", jpg(f, f"{r['id']}.right.jpg"), right[1]["filename"]))
        figures = "".join(f'<figure><figcaption>{html.escape(t)}</figcaption><a href="{src}"><img loading="lazy" src="{src}"></a>'
                          f'<small>{note}</small></figure>' for t, src, note in panels)
        cards.append(f'<section class="card {reviews[r["id"]]}"><h2>{html.escape(r["movie"].replace("_", " "))}'
                     f' <span class="tag">{names[reviews[r["id"]]]}</span>'
                     f'{" <span class=new>added in v2</span>" if r["cohort"] == "new" else ""}</h2>'
                     f'<div class="row">{figures}</div>'
                     f'<p class="src">{html.escape(r["id"])} &middot; <a href="{html.escape(q.get("page_url") or q["image_url"])}">source page</a>'
                     f'{"" if right else " &middot; right film not in the 20 results"}</p></section>')
    counts = {k: sum(reviews[r["id"]] == k for r in rows) for k in names}
    buttons = "".join(f'<button data-k="{k}">{names[k]} ({counts[k]})</button>' for k in names)
    (site / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Curtain misses</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--accent:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--accent:#f87171}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:16px}} h1{{margin:.2em 0}} p.lead{{color:var(--muted);margin-top:0}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 20px}}
button{{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--card);color:var(--fg);cursor:pointer}}
button.on{{border-color:var(--accent);color:var(--accent)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}}
.card h2{{font-size:17px;margin:0 0 8px}} .tag,.new{{font-size:12px;font-weight:500;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin-left:6px}}
.new{{color:var(--accent)}} .row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}
figure{{margin:0}} figcaption{{font-size:13px;font-weight:600;margin-bottom:4px}} img{{width:100%;border-radius:6px;display:block}}
small{{color:var(--muted)}} .src{{font-size:12px;color:var(--muted);margin:8px 0 0}} a{{color:inherit}}
</style></head><body><main><h1>Where the site picked the wrong film</h1>
<p class="lead">{len(rows)} of {sum(1 for r in map(json.loads, (args.out / 'results.jsonl').read_text().splitlines()) if r['variant'] == 'original')} Bing image-search results for 100 films, sent to the live site as a browser upload (640 px JPEG). Left: the web image. Middle: the site's first result. Right: the correct film's best frame, when it was in the 20 results.</p>
<nav><button data-k="all" class="on">All ({len(rows)})</button>{buttons}</nav>
{''.join(cards)}
</main><script>
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.card').forEach(c=>c.hidden=b.dataset.k!=='all'&&!c.classList.contains(b.dataset.k));}});
</script></body></html>""")
    print(f"{len(rows)} misses -> {site / 'index.html'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["collect", "sheets", "measure", "run", "report", "misses"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--new-films", type=int, default=60)
    p.add_argument("--v1-films", type=int, default=40)
    p.add_argument("--per-film", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--variants", nargs="+", choices=VARIANTS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
