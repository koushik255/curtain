from __future__ import annotations

import argparse
import html
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory, url_for
from PIL import UnidentifiedImageError

from src.model import DEFAULT_MODEL_PATH
from src.search import DEFAULT_INDEX, SearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = PROJECT_ROOT / "testing"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
engine: SearchEngine


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


@app.get("/")
def home():
    return send_file(PROJECT_ROOT / "web/index.html")


@app.get("/gallery")
def gallery():
    images = sorted(
        path
        for path in TESTING_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    items = []
    for path in images:
        name = html.escape(path.name)
        image_url = html.escape(url_for("gallery_image", filename=path.name), quote=True)
        items.append(
            f'<figure><a href="{image_url}"><img src="{image_url}" '
            f'alt="{name}" loading="lazy" width="320"></a>'
            f'<figcaption>{name}</figcaption></figure>'
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Curtain test gallery</title>
</head>
<body>
  <h1>Curtain test gallery</h1>
  <p><a href="/">Back to search</a></p>
  <p>{len(images)} test images. Click an image to open the original, then copy or save it.</p>
  {''.join(items)}
</body>
</html>"""
    return Response(page, mimetype="text/html")


@app.get("/gallery/image/<path:filename>")
def gallery_image(filename: str):
    return send_from_directory(TESTING_DIR, filename, conditional=True)


@app.get("/api/health")
def health():
    return jsonify(
        status="ready",
        movies=len(engine.shards),
        frames=engine.total_frames,
        sample_interval_seconds=1.0,
    )


@app.get("/frame/<int:shard_index>/<int:embedding_index>")
def frame_image(shard_index: int, embedding_index: int):
    record = engine.record(shard_index, embedding_index)
    if record is None:
        return jsonify(error="Frame not found."), 404
    path = engine.resolve_image_path(record).resolve()
    if not path.is_file():
        return jsonify(error="The indexed screenshot no longer exists."), 404
    return send_file(path, conditional=True, max_age=0)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="The image is larger than the 20 MB upload limit."), 413


@app.post("/api/search")
def search():
    data = request.get_data()
    if not data:
        return jsonify(error="Choose, drop, or paste an image first."), 400

    try:
        top_k = max(1, min(int(request.args.get("top_k", 5)), 10))
    except ValueError:
        return jsonify(error="top_k must be a number."), 400

    started = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        results = engine.search_bytes(data, top_k=top_k, timings=timings)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="That file is not a readable image."), 400

    public_results = []
    for result in results:
        public_results.append(
            {
                "movie_id": result["movie_id"],
                "title": result["title"],
                "year": result["year"],
                "frame_number": result["frame_number"],
                "timestamp_seconds": result["timestamp_seconds"],
                "score": result["score"],
                "image": f"/frame/{result['shard_index']}/{result['embedding_index']}",
            }
        )
    timings["server_total_ms"] = (time.perf_counter() - started) * 1000
    return jsonify(results=public_results, timings=timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Curtain screenshot search site.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    return parser.parse_args()


def main() -> None:
    global engine
    args = parse_args()
    engine = SearchEngine(args.index, args.model, args.device, preload=True)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
