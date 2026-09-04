from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file, stream_with_context, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

from training.train import CurtainEncoder, reference_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_models/curtain-resnet18-best.pt"
DEFAULT_INDEX = PROJECT_ROOT / "trained_indexes/The_Matrix_1999"
DEFAULT_FRAMES = PROJECT_ROOT / "screenshots/The_Matrix_1999"


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
        self.embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
        self.records = json.loads((index_dir / "records.json").read_text())
        self.frames_dir = frames_dir.resolve()
        if len(self.embeddings) != len(self.records):
            raise RuntimeError("Embedding and record counts do not match.")

    def search_bytes(self, data: bytes, top_k: int) -> tuple[list[dict], dict[str, float]]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            query = self.model(tensor)[0].float().cpu().numpy()
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

    def search_stream(self, data: bytes, chunk_size: int = 512) -> Iterator[dict]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        decoded = time.perf_counter()
        yield {"type": "stage", "stage": "decoded", "elapsed_ms": (decoded - started) * 1000}

        with torch.inference_mode():
            query = self.model(tensor)[0].float().cpu().numpy()
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

        record = dict(self.records[best_index])
        record.update(
            index=best_index,
            score=best_score,
            title="The Matrix",
            year=1999,
            image=url_for("frame_image", index=best_index),
        )
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
        candidate = (self.frames_dir / self.records[index]["filename"]).resolve()
        if candidate.parent != self.frames_dir or not candidate.is_file():
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
    return send_file(PROJECT_ROOT / "web/trained.html")


@app.get("/api/health")
def health():
    return jsonify(status="ready", movies=1, frames=len(engine.records), model="Curtain ResNet-18")


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
    for result in results:
        result["title"] = "The Matrix"
        result["year"] = 1999
        result["image"] = url_for("frame_image", index=result["index"])
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
    count = min(16, len(engine.records))
    indices = np.linspace(0, len(engine.records) - 1, num=count, dtype=int)
    figures = "".join(
        f'<figure><a href="{url_for("frame_image", index=int(index))}">'
        f'<img src="{url_for("frame_image", index=int(index))}" loading="lazy" width="320" '
        f'alt="The Matrix frame {engine.records[index]["frame_number"]}"></a>'
        f'<figcaption>{engine.records[index]["filename"]}</figcaption></figure>'
        for index in indices
    )
    return Response(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Curtain trained-model test gallery</title></head><body>"
        '<h1>Curtain trained-model test gallery</h1><p><a href="/">Back to search</a></p>'
        "<p>These frames are spread across The Matrix. Save, copy, or drag one into the search page.</p>"
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
