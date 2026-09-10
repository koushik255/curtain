from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from curtain_ml.training import CurtainEncoder, reference_transform
from curtain_ml.retrieval import normalize_rows


def search(
    query_path: Path,
    checkpoint_path: Path,
    index_dir: Path,
    top_k: int = 10,
    device_name: str = "auto",
) -> list[dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device_name = (
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    model = CurtainEncoder(int(config["embedding_dim"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    with Image.open(query_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    query = reference_transform(int(config["image_size"]))(image).unsqueeze(0)
    with torch.inference_mode():
        query_embedding = normalize_rows(model(query.to(device)).float().cpu().numpy())[0]

    embeddings = normalize_rows(np.load(index_dir / "embeddings.npy"))
    records = json.loads((index_dir / "records.json").read_text())
    if len(embeddings) != len(records):
        raise RuntimeError("The embedding matrix and records file have different lengths.")

    # Re-normalize after float16 storage rounding so the dot product is cosine.
    scores = embeddings @ query_embedding
    count = min(max(top_k, 1), len(scores))
    best = np.argpartition(scores, -count)[-count:]
    best = best[np.argsort(scores[best])[::-1]]
    return [dict(records[index], score=float(scores[index])) for index in best]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a trained Curtain movie index.")
    parser.add_argument("query", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    matches = search(
        query_path=args.query.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        index_dir=args.index.expanduser().resolve(),
        top_k=args.top_k,
        device_name=args.device,
    )
    print(json.dumps(matches, indent=2))
