"""Measure retrieval robustness against the active index."""

import argparse
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance

from apps.server import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FRAMES,
    DEFAULT_INDEX,
    TrainedSearchEngine,
    app,
)


def alter(image: Image.Image, condition: str) -> Image.Image:
    # Apply one controlled robustness condition to a reference frame.
    if condition == "dark":
        return ImageEnhance.Brightness(image).enhance(0.55)
    if condition == "crop":
        width, height = image.size
        return image.crop((width * 0.08, height * 0.08, width * 0.92, height * 0.92))
    return image


def main() -> None:
    # Sample indexed frames and report movie/frame retrieval accuracy.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--condition", choices=("original", "dark", "crop"), default="crop")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT, DEFAULT_INDEX, DEFAULT_FRAMES)
    generator = random.Random(args.seed)
    selected = []
    while len(selected) < min(args.samples, len(engine.records)):
        index = generator.randrange(len(engine.records))
        path = engine.frame_path(index)
        if path is not None and all(saved[0] != index for saved in selected):
            selected.append((index, path))
    exact = movie = 0
    for index, path in selected:
        source = engine.records[index]
        with Image.open(path) as image:
            query = alter(image.convert("RGB"), args.condition)
            encoded = io.BytesIO()
            query.save(encoded, format="JPEG", quality=82)
        with app.test_request_context():
            result = list(engine.search_stream(encoded.getvalue()))[-1]["result"]
        exact += result["movie"] == source["movie"] and result["filename"] == source["filename"]
        movie += result["movie"] == source["movie"]

    report = {
        "condition": args.condition,
        "samples": len(selected),
        "exact_frame_top1": exact / len(selected),
        "movie_top1": movie / len(selected),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
