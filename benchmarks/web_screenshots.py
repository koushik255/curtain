"""Audit independently sourced web screenshots against the running site.

    python -m benchmarks.web_screenshots fetch
    python -m benchmarks.web_screenshots run

Original downloads stay unchanged and are tested directly. Re-encoded variants
start from the same decoded, 640px image: browser-like JPEG 90, JPEG 20, or WebP 30. These are additional
image compressions, not evidence of the source video's codec. Movie labels
come from the source pages; exact frames are unknown unless manually labelled
with a `target` (collection/movie/filename.jpg) in the manifest. A geometric
confirmation is reported separately and is not an exact-frame ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/data/web_screenshots_v1.json"
DEFAULT_OUT = ROOT / ".local/benchmarks/web-screenshots-v1"
VARIANTS = {"original": None, "browser": ("JPEG", 90), "jpeg20": ("JPEG", 20), "webp30": ("WEBP", 30)}


def request(url: str, data: bytes | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Curtain screenshot evaluation)",
        **({"Content-Type": "application/octet-stream"} if data is not None else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def fetch(manifest: dict, out: Path) -> None:
    originals = out / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in manifest["queries"]:
        path = originals / case["id"]
        try:
            data = path.read_bytes() if path.exists() else request(case["image_url"], timeout=20)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                info = {"format": image.format, "width": image.width, "height": image.height}
            path.write_bytes(data)
            rows.append({"id": case["id"], "sha256": hashlib.sha256(data).hexdigest(),
                         "bytes": len(data), **info})
            print(f"{case['id']}: {info['format']} {info['width']}x{info['height']}", flush=True)
        except (OSError, ValueError) as exc:
            rows.append({"id": case["id"], "error": str(exc)})
            print(f"{case['id']}: download failed: {exc}", flush=True)
        # Persist progress if a later site stalls or the run is interrupted.
        save_json(out / "downloads.json", rows)


def payload(path: Path, variant: str) -> bytes:
    if variant == "original":
        return path.read_bytes()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = min(1.0, 640 / max(image.size))
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BILINEAR)
    format, quality = VARIANTS[variant]
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def run(manifest: dict, out: Path, base_url: str, variants: list[str]) -> None:
    movies = read_json_bytes(request(base_url + "/api/movies"))["movies"]
    available = {m["id"] for m in movies}
    health = read_json_bytes(request(base_url + "/api/health"))
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "base_url": base_url, "health": health,
              "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
              "sampling": manifest["sampling"], "rows": []}
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    report_path = out / f"results-{stamp}.json"
    for case in manifest["queries"]:
        path = out / "originals" / case["id"]
        for variant in variants:
            row = {**case, "variant": variant, "indexed": case["movie"] in available}
            try:
                if not path.exists():
                    raise FileNotFoundError(f"Missing download: {path}")
                data = payload(path, variant)
                events = [json.loads(line) for line in request(base_url + "/api/search-stream", data).splitlines()
                          if line.strip()]
                errors = [e for e in events if e.get("type") == "error"]
                result = next((e for e in reversed(events) if e.get("type") == "result"), None)
                if errors or result is None:
                    raise ValueError(str(errors or "No result event"))
                found = result.get("results", [])
                top = found[0] if found else {}
                correct_movie = top.get("movie") == case["movie"]
                target = case.get("target")
                row.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           payload_sha256=hashlib.sha256(data).hexdigest(), payload_bytes=len(data),
                           movie_top1=correct_movie,
                           movie_top5=any(r["movie"] == case["movie"] for r in found[:5]),
                           exact_top1=(f"{top.get('source')}/{top.get('filename')}" == target) if target else None,
                           confirmed_correct_movie=correct_movie and bool(top.get("confirmed")),
                           weak=result.get("weak"), result=top, results=found, timings=result.get("timings"))
                status = "right movie" if correct_movie else "WRONG MOVIE"
                print(f"{case['id']} {variant}: {status}, {top.get('points', 0)} points, "
                      f"{top.get('movie')} / {top.get('filename')}", flush=True)
            except (OSError, ValueError) as exc:
                row["error"] = str(exc)
                print(f"{case['id']} {variant}: ERROR {exc}", flush=True)
            report["rows"].append(row)
            save_json(report_path, report)
    write_report(report_path)
    print(f"Saved {report_path}", flush=True)


def read_json_bytes(data: bytes):
    return json.loads(data)


def write_report(path: Path) -> None:
    report = read_json(path)
    rows = report["rows"]
    lines = ["# Independent web screenshot audit", "", report["sampling"], "",
             "These are a small convenience sample, not an accuracy estimate for arbitrary image searches.",
             "Exact frames are unlabelled unless explicitly supplied. SIFT confirmation can also match neighbouring frames.",
             "Original = unchanged download sent directly to the API. Browser = 640px JPEG 90 approximation of browser upload.",
             "The real browser may keep the original when smaller, and uses its own resize/JPEG implementation.",
             "JPEG 20 and WebP 30 add compression to the same web image at the same size.",
             "These quality numbers are not matched bitrate or matched visual quality, and do not identify source video codecs.", "",
             "| Variant | Completed indexed queries | Right movie #1 | Right movie top 5 | Geometrically confirmed, right movie |",
             "|---|---:|---:|---:|---:|"]
    for variant in dict.fromkeys(r["variant"] for r in rows):
        sub = [r for r in rows if r["variant"] == variant and r["indexed"] and "error" not in r]
        lines.append(f"| {variant} | {len(sub)} | {sum(r['movie_top1'] for r in sub)} | "
                     f"{sum(r['movie_top5'] for r in sub)} | {sum(r['confirmed_correct_movie'] for r in sub)} |")
    originals = {r["id"]: r for r in rows if r["variant"] == "original" and r["indexed"]
                 and r.get("movie_top1")}
    if originals:
        lines += ["", "## Frame selection stability", "",
                  "Among queries finding the right movie both originally and after re-encoding.",
                  "Agreement with the original result is stability, not exact-frame accuracy.", "",
                  "| Variant | Comparable queries | Same top frame as original |", "|---|---:|---:|"]
        for variant in dict.fromkeys(r["variant"] for r in rows if r["variant"] != "original"):
            sub = [r for r in rows if r["variant"] == variant and r["id"] in originals and r.get("movie_top1")]
            same = sum((r["result"]["source"], r["result"]["filename"]) ==
                       (originals[r["id"]]["result"]["source"], originals[r["id"]]["result"]["filename"])
                       for r in sub)
            lines.append(f"| {variant} | {len(sub)} | {same} |")
    lines += ["", "## Individual results", "",
              "| Query | Variant | Indexed | Right movie | SIFT points | Returned frame |",
              "|---|---|---|---|---:|---|"]
    for r in rows:
        result = r.get("result", {})
        returned = r.get("error", f"{result.get('movie', '')}/{result.get('filename', '')}").replace("|", "\\|")
        lines.append(f"| [{r['id']}]({r['page_url']}) | {r['variant']} | {r['indexed']} | "
                     f"{r.get('movie_top1', 'error')} | {result.get('points', '')} | {returned} |")
    missing = [r for r in rows if not r["indexed"] and "error" not in r]
    lines += ["", f"Missing-movie controls: {len(missing)} searches; "
              f"{sum(not r.get('weak', True) for r in missing)} returned without a weak-match flag; "
              f"{sum(bool(r.get('result', {}).get('confirmed')) for r in missing)} geometrically confirmed.",
              f"Errors: {sum('error' in r for r in rows)} (excluded from completed-query counts).", ""]
    path.with_suffix(".md").write_text("\n".join(lines))


def contact_sheet(report_path: Path, base_url: str) -> None:
    """Query beside top returned frame, for manual review (browser variant)."""
    rows = [r for r in read_json(report_path)["rows"] if r["variant"] == "browser" and "error" not in r]
    if not rows:
        return
    width, height = 680, 245
    canvas = Image.new("RGB", (width * 2, height * ((len(rows) + 1) // 2)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        left, top = (i % 2) * width, (i // 2) * height
        with Image.open(report_path.parent / "originals" / row["id"]) as source:
            query = ImageOps.exif_transpose(source).convert("RGB")
        try:
            frame_data = request(base_url + row["result"]["image"])
            with Image.open(io.BytesIO(frame_data)) as source:
                frame = source.convert("RGB")
        except (OSError, ValueError):
            frame = Image.new("RGB", (320, 180), "#333333")
        for x, image in [(left, query), (left + 335, frame)]:
            fitted = ImageOps.contain(image, (325, 180))
            canvas.paste(fitted, (x, top + 24))
        color = "#78e49a" if row["movie_top1"] else "#ff9292"
        draw.text((left, top + 4), row["id"], fill=color)
        result = row["result"]
        draw.text((left, top + 208), f"{result['movie']}  {result['filename']}", fill=color)
        draw.text((left, top + 225), f"SIFT: {result.get('points', 0)}  | exact frame needs manual verification", fill="white")
    canvas.save(report_path.with_suffix(".jpg"), quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "run", "report", "sheet"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8781")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        fetch(read_json(args.manifest), args.out)
    elif args.command == "run":
        run(read_json(args.manifest), args.out, args.base_url.rstrip("/"), args.variants)
    elif args.results is None:
        parser.error("--results is required for report and sheet")
    elif args.command == "report":
        write_report(args.results)
    else:
        contact_sheet(args.results, args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
