from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from curtain_ml.training import CurtainEncoder, IMAGE_SUFFIXES, reference_transform


class MovieFrameDataset(Dataset):
    def __init__(self, movie_dir: Path, image_size: int):
        self.paths = sorted(
            path for path in movie_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {movie_dir}.")
        self.transform = reference_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), path.name


def frame_number(filename: str) -> int | None:
    stem = Path(filename).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def build_movie_index(
    checkpoint_path: Path,
    movie_dir: Path,
    output_dir: Path,
    batch_size: int = 256,
    workers: int = 8,
    device_name: str = "auto",
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    movie_dir = movie_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    image_size = int(config["image_size"])
    embedding_dim = int(config["embedding_dim"])
    device_name = (
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    model = CurtainEncoder(embedding_dim)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset = MovieFrameDataset(movie_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    started = time.perf_counter()
    batches: list[np.ndarray] = []
    filenames: list[str] = []
    with torch.inference_mode():
        for images, names in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                embeddings = model(images.to(device, non_blocking=True))
            batches.append(embeddings.float().cpu().numpy())
            filenames.extend(names)

    embeddings = np.concatenate(batches).astype(np.float16)
    np.save(output_dir / "embeddings.npy", embeddings)
    records = []
    for filename in filenames:
        number = frame_number(filename)
        records.append(
            {
                "movie": movie_dir.name,
                "filename": filename,
                "frame_number": number,
                "timestamp_seconds": number / 2.0 if number is not None else None,
            }
        )
    (output_dir / "records.json").write_text(json.dumps(records) + "\n")
    manifest = {
        "movie": movie_dir.name,
        "frames": len(records),
        "dimensions": embedding_dim,
        "dtype": "float16",
        "image_size": image_size,
        "checkpoint": checkpoint_path.name,
        "encoding_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a complete movie with Curtain's trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_movie_index(
        checkpoint_path=args.checkpoint,
        movie_dir=args.movie,
        output_dir=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )
