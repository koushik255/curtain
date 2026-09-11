"""Command-line entry point for the standalone experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gpu_video_pipeline.pipeline import index_movie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a movie directly through NVDEC and a TorchScript encoder."
    )
    parser.add_argument("movie", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="TorchScript encoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = torch.jit.load(str(args.model), map_location="cuda")
    manifest = index_movie(
        movie=args.movie,
        model=model,
        output_dir=args.output,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
