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

from curtain_ml.training import CurtainEncoder, reference_transform
from curtain_ml.retrieval import normalize_rows
from curtain_ml.provenance import LABELS, EXPLANATION, category, load_split, movie_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_models/curtain-resnet18-50k-best.pt"
DEFAULT_INDEX = PROJECT_ROOT / "trained_indexes/l4-50k"
DEFAULT_FRAMES = PROJECT_ROOT / "screenshots"


class TrainedSearchEngine:
    def __init__(self, checkpoint: Path, index_dir: Path, frames_dir: Path, device: str):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = saved["config"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CurtainEncoder(int(config["embedding_dim"]))
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()
        self.transform = reference_transform(int(config["image_size"]))
        self.frames_dir = frames_dir.resolve()
        collection = json.loads((index_dir / "collection.json").read_text())
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if collection["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Collection and model checkpoint do not match.")
        split = load_split(checkpoint, checkpoint_hash, PROJECT_ROOT)
        self.records = []
        self.movies = []
        matrices = []
        for entry in collection["movies"]:
            name = entry["movie"]
            if Path(name).name != name:
                raise RuntimeError("Invalid movie directory.")
            folder = index_dir / name
            manifest = json.loads((folder / "manifest.json").read_text())
            records = json.loads((folder / "records.json").read_text())
            matrix = np.load(folder / "embeddings.npy").astype(np.float32)
            if (manifest["checkpoint_sha256"] != checkpoint_hash
                    or matrix.shape != (entry["frames"], int(config["embedding_dim"]))
                    or len(records) != len(matrix) or not np.isfinite(matrix).all()
                    or any(record["movie"] != name for record in records)):
                raise RuntimeError(f"Invalid index for {name}.")
            title, separator, year = name.rpartition("_")
            if not separator or not year.isdigit():
                title, year = name, ""
            self.movies.append(dict(id=name, title=title.replace("_", " "),
                                    year=int(year) if year else None, frames=len(records),
                                    training_status=category(name, split),
                                    training_label=LABELS[category(name, split)],
                                    start=len(self.records)))
            self.records.extend(records)
            matrices.append(normalize_rows(matrix))
        if not matrices or len(self.records) != collection["total_frames"]:
            raise RuntimeError("Collection frame count does not match.")
        self.embeddings = np.concatenate(matrices)
        self.movie_lookup = {movie["id"]: movie for movie in self.movies}
        self.source_dirs = {}
        for movie in self.movies:
            movie['spans'] = [(movie['start'], movie['frames'])]
            self.source_dirs[movie['id']] = (self.frames_dir, movie['id'])
        self.catalog = [dict(m, active=True) for m in self.movies]
        known = {movie_key(m['id']): m for m in self.movies}
        extra_path = PROJECT_ROOT / 'trained_indexes/l4-50k-lbfive/collection.json'
        if extra_path.exists():
            extra = json.loads(extra_path.read_text())
            if index_dir.resolve() == DEFAULT_INDEX.resolve():
                if extra.get('checkpoint_sha256') != checkpoint_hash:
                    raise RuntimeError('Additional collection and model checkpoint do not match')
                extra_root = Path('/home/koushik/lbfive/screenshots').resolve()
                extra_total = 0
                extra_matrices = [self.embeddings]
                for item in extra['movies']:
                    name = item['movie']
                    if Path(name).name != name:
                        raise RuntimeError('Invalid extra movie directory')
                    folder = extra_path.parent / name
                    manifest = json.loads((folder / 'manifest.json').read_text())
                    records = json.loads((folder / 'records.json').read_text())
                    matrix = np.load(folder / 'embeddings.npy')
                    if (manifest.get('checkpoint_sha256') != checkpoint_hash
                            or matrix.shape != (item['frames'], int(config['embedding_dim']))
                            or len(records) != len(matrix)
                            or any(r['movie'] != name or Path(r['filename']).name != r['filename'] for r in records)):
                        raise RuntimeError(f'Invalid additional index: {name}')
                    source = 'lbfive:' + name
                    self.source_dirs[source] = (extra_root, name)
                    start = len(self.records)
                    title, _, year = name.rpartition('_')
                    status = category(name, split)
                    movie = known.get(movie_key(name))
                    if movie is None:
                        movie = dict(id=name, title=title.replace('_', ' '),
                            year=int(year) if year.isdigit() else None, frames=0, start=start,
                            spans=[], training_status=status, training_label=LABELS[status])
                        self.movies.append(movie)
                        self.movie_lookup[name] = movie
                        known[movie_key(name)] = movie
                    movie['spans'].append((start, len(records)))
                    movie['frames'] += len(records)
                    self.records.extend(dict(r, movie=movie['id'], source=source) for r in records)
                    extra_matrices.append(normalize_rows(matrix))
                    extra_total += len(records)
                if extra_total != extra['total_frames']:
                    raise RuntimeError('Additional collection frame count does not match')
                self.embeddings = np.concatenate(extra_matrices)
                self.catalog = [dict(m, active=True) for m in self.movies]

    def public_result(self, index: int, score: float) -> dict:
        record = dict(self.records[index])
        movie = self.movie_lookup[record["movie"]]
        record.update(index=index, score=score, title=movie["title"], year=movie["year"],
                      training_status=movie['training_status'], training_label=movie['training_label'],
                      image=url_for("frame_image", index=index))
        return record

    def search_bytes(self, data: bytes, top_k: int) -> tuple[list[dict], dict[str, float]]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            query = normalize_rows(self.model(tensor).float().cpu().numpy())[0]
        embedded = time.perf_counter()
        scores = self.embeddings @ query
        count = min(top_k, len(scores))
        best = np.argpartition(scores, -count)[-count:]
        best = best[np.argsort(scores[best])[::-1]]
        compared = time.perf_counter()
        results = [dict(self.records[index], score=float(scores[index]), index=int(index)) for index in best]
        return results, {
            "embedding_ms": (embedded - started) * 1000,
            "comparison_ms": (compared - embedded) * 1000,
            "server_total_ms": (compared - started) * 1000,
        }

    def search_stream(self, data: bytes, chunk_size: int = 8192) -> Iterator[dict]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        decoded = time.perf_counter()
        yield {"type": "stage", "stage": "decoded", "elapsed_ms": (decoded - started) * 1000}

        with torch.inference_mode():
            query = normalize_rows(self.model(tensor).float().cpu().numpy())[0]
        embedded = time.perf_counter()
        yield {
            "type": "stage",
            "stage": "embedded",
            "elapsed_ms": (embedded - started) * 1000,
            "embedding_ms": (embedded - decoded) * 1000,
        }

        best_index = -1
        best_score = float("-inf")
        total = len(self.embeddings)
        search_started = time.perf_counter()
        for offset in range(0, total, chunk_size):
            end = min(offset + chunk_size, total)
            scores = self.embeddings[offset:end] @ query
            local_index = int(np.argmax(scores))
            local_score = float(scores[local_index])
            if local_score > best_score:
                best_score = local_score
                best_index = offset + local_index
            elapsed = time.perf_counter() - search_started
            yield {
                "type": "progress",
                "stage": "searching",
                "processed": end,
                "total": total,
                "percent": end / total * 100,
                "retrieval_ms": elapsed * 1000,
                "frames_per_second": end / elapsed if elapsed else 0,
            }

        record = self.public_result(best_index, best_score)
        finished = time.perf_counter()
        yield {
            "type": "result",
            "result": record,
            "timings": {
                "decode_ms": (decoded - started) * 1000,
                "embedding_ms": (embedded - decoded) * 1000,
                "comparison_ms": (finished - search_started) * 1000,
                "server_total_ms": (finished - started) * 1000,
            },
        }

    def frame_path(self, index: int) -> Path | None:
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
    return send_file(PROJECT_ROOT / "apps/templates/index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ready", movies=len(engine.movies), frames=len(engine.records),
                   model="Curtain ResNet-18 · 50k")


@app.get("/api/movies")
def movies():
    return jsonify(movies=[dict(id=m["id"], title=m["title"], year=m["year"],
                               training_status=m['training_status'], training_label=m['training_label'],
                               frames=m["frames"], gallery=url_for("gallery", movie=m["id"]))
                           for m in engine.movies])


@app.get('/movies')
def movie_catalog():
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
        f'<p>Labels refer to the active 50k model.</p><p>{html.escape(EXPLANATION)}</p>'
        '<p>Unseen does not mean unindexed: indexing does not train the model. '
        'Only movies marked searchable now can return matches on this site.</p>'
        + ''.join(sections) + '</body></html>', mimetype='text/html')


@app.post("/api/search")
def search():
    data = request.get_data()
    if not data:
        return jsonify(error="Choose, drop, or paste an image first."), 400
    try:
        top_k = max(1, min(int(request.args.get("top_k", 5)), 10))
        results, timings = engine.search_bytes(data, top_k)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="That file is not a readable image."), 400
    results = [engine.public_result(r["index"], r["score"]) for r in results]
    return jsonify(results=results, timings=timings)


@app.post("/api/search-stream")
def search_stream():
    data = request.get_data()
    if not data:
        return jsonify(error="Choose, drop, or paste an image first."), 400

    @stream_with_context
    def generate():
        try:
            for event in engine.search_stream(data):
                yield json.dumps(event, separators=(",", ":")) + "\n"
        except (UnidentifiedImageError, OSError, ValueError):
            yield json.dumps({"type": "error", "error": "That file is not a readable image."}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/frame/<int:index>")
def frame_image(index: int):
    path = engine.frame_path(index)
    if path is None:
        return jsonify(error="Frame not found."), 404
    return send_file(path, conditional=True, max_age=0)


@app.get("/gallery")
def gallery():
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
            for start, length in movie.get('spans', [(movie['start'], movie['frames'])]):
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
    return jsonify(error="The image is larger than the 20 MB upload limit."), 413


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the trained Curtain image search site.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    return parser.parse_args()


def main() -> None:
    global engine
    args = parse_args()
    engine = TrainedSearchEngine(args.checkpoint, args.index, args.frames, args.device)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
