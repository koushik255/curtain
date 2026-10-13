from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import time
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file, stream_with_context, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import reference_transform
from curtain_ml.retrieval import normalize_rows
from curtain_ml.provenance import LABELS, EXPLANATION, category, load_split, movie_key
from rcmindex import search as tile_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = PROJECT_ROOT / ".local"
DEFAULT_CHECKPOINT = LOCAL_ROOT / "trained_models/combined-rocm-50k-v1/best.pt"
# Frame folders per index collection, and the older per-frame records that
# carry timestamps.
FRAME_ROOTS = {
    "full-curtain": LOCAL_ROOT / "screenshots",
    "l4-50k-lbfive": Path("/home/koushik/lbfive/screenshots"),
}
TIMESTAMP_RECORDS = {
    "full-curtain": LOCAL_ROOT / "trained_indexes/l4-50k-crop-v1",
    "l4-50k-lbfive": LOCAL_ROOT / "trained_indexes/l4-50k-lbfive-crop-v1",
}
# Below this, the best match is shown with a low-confidence note. On the crop
# benchmark it flags ~45% of wrong-movie answers and ~11% of right ones.
WEAK_SCORE = 0.70
RESULTS = 5


class TrainedSearchEngine:
    def __init__(self, checkpoint: Path):
        # Load the model, the FAISS tile index and per-frame records.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CurtainEncoder.from_checkpoint(checkpoint, self.device)
        self.transform = reference_transform(self.model.image_size)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        split = load_split(checkpoint, checkpoint_hash, PROJECT_ROOT)
        self.search_index = tile_search.load()

        self.records = []
        self.movies = []
        self.movie_lookup = {}
        self.source_dirs = {}
        known = {}
        catalog = json.loads((tile_search.FAISS_DIR / "catalog.json").read_text())
        names = catalog["names"]  # "<collection>/<movie>/<frame>.jpg" in frame-id order
        for entry in catalog["movies"]:
            collection, name = entry["movie"].split("/")
            if Path(name).name != name or collection not in FRAME_ROOTS:
                raise RuntimeError(f"Invalid movie entry {entry['movie']}.")
            start, count = entry["first"], entry["frames"]
            if start != len(self.records):
                raise RuntimeError("Catalog frame ids are not contiguous.")
            source = entry["movie"]
            self.source_dirs[source] = (FRAME_ROOTS[collection].resolve(), name)
            timestamps = {}
            old = TIMESTAMP_RECORDS[collection] / name / "records.json"
            if old.exists():
                timestamps = {r["filename"]: r["timestamp_seconds"] for r in json.loads(old.read_text())}

            key = movie_key(name)
            movie = known.get(key)
            if movie is None:
                title, separator, year = name.rpartition("_")
                if not separator or not year.isdigit():
                    title, year = name, ""
                status = category(name, split)
                movie = {
                    "id": name,
                    "title": title.replace("_", " "),
                    "year": int(year) if year else None,
                    "frames": 0,
                    "spans": [],
                    "training_status": status,
                    "training_label": LABELS[status],
                }
                known[key] = movie
                self.movies.append(movie)
                self.movie_lookup[name] = movie
            movie["spans"].append((start, count))
            movie["frames"] += count
            for full in names[start:start + count]:
                filename = full.rsplit("/", 1)[1]
                if Path(filename).name != filename:
                    raise RuntimeError(f"Invalid frame name {full}.")
                self.records.append({
                    "movie": movie["id"],
                    "source": source,
                    "filename": filename,
                    "frame_number": int(Path(filename).stem.rsplit("_", 1)[1]),
                    "timestamp_seconds": timestamps.get(filename),
                })
        if not self.records:
            raise RuntimeError("No frames loaded.")
        self.vectors = self.search_index["index"].ntotal
        self.catalog = [dict(movie, active=True) for movie in self.movies]

    def public_result(self, index: int, score: float) -> dict:
        # Convert an internal matrix row into the JSON shown by the web UI.
        record = dict(self.records[index])
        movie = self.movie_lookup[record["movie"]]
        record.update(index=index, score=score, title=movie["title"], year=movie["year"],
                      training_status=movie['training_status'], training_label=movie['training_label'],
                      image=url_for("frame_image", index=index))
        return record

    def search_stream(self, data: bytes) -> Iterator[dict]:
        # Yield decode, embedding, search, and final-result events for one query.
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        # The whole query plus four overlapping pieces: a crop that covers only
        # part of a frame still matches that frame's tiles piece by piece.
        views = [image, *tile_search.pieces(image)]
        tensor = torch.stack([self.transform(view) for view in views]).to(self.device)
        decoded = time.perf_counter()
        yield {"type": "stage", "stage": "decoded", "elapsed_ms": (decoded - started) * 1000}

        with torch.inference_mode():
            queries = normalize_rows(self.model(tensor).float().cpu().numpy())
        embedded = time.perf_counter()
        yield {
            "type": "stage",
            "stage": "embedded",
            "elapsed_ms": (embedded - started) * 1000,
            "embedding_ms": (embedded - decoded) * 1000,
        }

        found = tile_search.search(self.search_index, queries, top=RESULTS)
        finished = time.perf_counter()
        yield {
            "type": "progress",
            "stage": "searching",
            "processed": self.vectors,
            "total": self.vectors,
            "percent": 100.0,
            "retrieval_ms": (finished - embedded) * 1000,
            "frames_per_second": len(self.records) / (finished - embedded),
        }
        results = [self.public_result(index, score) for index, score in found]
        yield {
            "type": "result",
            "result": results[0] if results else None,
            "results": results,
            "weak": not results or results[0]["score"] < WEAK_SCORE,
            "timings": {
                "decode_ms": (decoded - started) * 1000,
                "embedding_ms": (embedded - decoded) * 1000,
                "comparison_ms": (finished - embedded) * 1000,
                "server_total_ms": (finished - started) * 1000,
            },
        }

    def frame_path(self, index: int) -> Path | None:
        # Resolve a safe local source path for an indexed frame row.
        if not 0 <= index < len(self.records):
            return None
        record = self.records[index]
        root, folder = self.source_dirs[record.get('source', record['movie'])]
        movie_dir = (root / folder).resolve()
        candidate = (movie_dir / record["filename"]).resolve()
        if (movie_dir.parent != root or candidate.parent != movie_dir
                or not candidate.is_file()):
            return None
        return candidate


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
engine: TrainedSearchEngine


@app.after_request
def security_headers(response):
    # Add conservative headers to every response from the site.
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
    # Serve the single-page search interface.
    return send_file(PROJECT_ROOT / "apps/templates/index.html")


@app.get("/api/health")
def health():
    # Return a lightweight readiness and index-size check.
    return jsonify(status="ready", movies=len(engine.movies), frames=len(engine.records),
                   vectors=engine.vectors, model="Curtain ResNet-18 · combined tiles")


@app.get("/api/movies")
def movies():
    # Return searchable movie metadata for the gallery UI.
    return jsonify(movies=[dict(id=m["id"], title=m["title"], year=m["year"],
                               training_status=m['training_status'], training_label=m['training_label'],
                               frames=m["frames"], gallery=url_for("gallery", movie=m["id"]))
                           for m in engine.movies])


@app.get('/movies')
def movie_catalog():
    # Render the training/validation/unseen movie catalog as HTML.
    sections = []
    for status, label in LABELS.items():
        rows = sorted((m for m in engine.catalog if m['training_status'] == status), key=lambda m:m['title'].casefold())
        items = []
        for m in rows:
            title = html.escape(f"{m['title']} ({m['year']})")
            if m['active']:
                title = f'<a href="{url_for("gallery", movie=m["id"])}">{title}</a> — searchable now'
            else:
                title += ' — indexed separately; not searchable on this site yet'
            items.append(f'<li>{title}</li>')
        sections.append(f'<h2>{html.escape(label)} ({len(rows)})</h2><ul>{"".join(items)}</ul>')
    return Response('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Movie training history</title>'
        '</head><body><h1>Movie training history</h1><p><a href="/">Back to search</a></p>'
        f'<p>Labels refer to the active combined tile model.</p><p>{html.escape(EXPLANATION)}</p>'
        '<p>Unseen does not mean unindexed: indexing does not train the model. '
        'Only movies marked searchable now can return matches on this site.</p>'
        + ''.join(sections) + '</body></html>', mimetype='text/html')


@app.post("/api/search-stream")
def search_stream():
    # Accept an image upload and stream newline-delimited search progress.
    data = request.get_data()
    if not data:
        return jsonify(error="Choose, drop, or paste an image first."), 400

    @stream_with_context
    def generate():
        # Keep the Flask request context while forwarding engine events.
        try:
            for event in engine.search_stream(data):
                yield json.dumps(event, separators=(",", ":")) + "\n"
        except (UnidentifiedImageError, OSError, ValueError):
            yield json.dumps({"type": "error", "error": "That file is not a readable image."}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/frame/<int:index>")
def frame_image(index: int):
    # Serve one validated source frame by its internal index row.
    path = engine.frame_path(index)
    if path is None:
        return jsonify(error="Frame not found."), 404
    return send_file(path, conditional=True, max_age=0)


@app.get("/gallery")
def gallery():
    # Render sample frames with movie and provenance filters.
    selected = request.args.get("movie")
    group = request.args.get("group", "all")
    groups = {"all": "All movies", "training": "Training movies", "validation": "Validation movies",
              "unseen": "Unseen by training/validation"}
    if group not in groups:
        return jsonify(error="Unknown image group."), 400
    if selected and selected not in engine.movie_lookup:
        return jsonify(error="Movie not found."), 404
    movie_list = [engine.movie_lookup[selected]] if selected else engine.movies
    if group != "all":
        movie_list = [movie for movie in movie_list if movie['training_status'] == group]
    options = ''.join(f'<option value="{key}"{" selected" if key == group else ""}>{label}</option>'
                      for key, label in groups.items())
    filters = ('<form action="/gallery" method="get"><label for="image-group">Show images from: </label>'
               f'<select id="image-group" name="group">{options}</select> '
               '<button type="submit">Apply</button></form>'
               '<p>Groups refer to the movie’s training history, not whether each individual frame was used.</p>')
    indices = []
    for movie in movie_list:
        count = min(16 if selected else 2, movie["frames"])
        for position in np.linspace(0, movie['frames'] - 1, count + 2, dtype=int)[1:-1]:
            spans = movie.get('spans')
            if spans is None:
                spans = [(movie['start'], movie['frames'])]
            for start, length in spans:
                if position < length:
                    indices.append(start + position)
                    break
                position -= length
    figures = "".join(
        f'<figure><a href="{url_for("frame_image", index=int(index))}">'
        f'<img src="{url_for("frame_image", index=int(index))}" loading="lazy" width="320" '
        f'alt="{html.escape(engine.movie_lookup[engine.records[index]["movie"]]["title"], quote=True)} '
        f'frame {engine.records[index]["frame_number"]}"></a>'
        f'<figcaption>{html.escape(engine.movie_lookup[engine.records[index]["movie"]]["title"])} '
        f'[{html.escape(engine.movie_lookup[engine.records[index]["movie"]]["training_label"])}] '
        f'— {html.escape(engine.records[index]["filename"])}</figcaption></figure>'
        for index in indices
    )
    return Response(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Curtain trained-model test gallery</title></head><body>"
        '<h1>Curtain trained-model test gallery</h1><p><a href="/">Back to search</a></p>'
        '<p><a href="/gallery">All movies</a></p>'
        f'<p><a href="/movies">Movie training history and unseen movie list</a></p><p>{html.escape(EXPLANATION)}</p>'
        "<p>Examples from the indexed movies. Save or copy an image to test it on the search page.</p>"
        f'{filters}<p>{len(movie_list)} movies · {len(indices)} sample images</p>'
        + ('<p>No images in this group.</p>' if not indices else '') +
        f"{figures}</body></html>",
        mimetype="text/html",
    )


@app.errorhandler(413)
def too_large(_error):
    # Explain upload-size failures in the same JSON format as other errors.
    return jsonify(error="The image is larger than the 20 MB upload limit."), 413


def parse_args() -> argparse.Namespace:
    # Parse the host and port for the local web process.
    parser = argparse.ArgumentParser(description="Serve the trained Curtain image search site.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    return parser.parse_args()


def main() -> None:
    # Load the fixed active artifacts and start the Flask server.
    global engine
    args = parse_args()
    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
