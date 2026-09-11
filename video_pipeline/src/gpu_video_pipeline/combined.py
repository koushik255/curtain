"""One-pass GPU indexing: NVDEC decode -> JPEG screenshots + embeddings.

Each sampled frame is decoded once into CUDA memory, then reused for both
outputs:

  * resize to at most ``--screenshot-width`` and JPEG-encode on the GPU
    (nvJPEG via torchvision) into ``<screenshots>/<movie>/frame_%06d.jpg``;
  * resize to the encoder input and run the model for embeddings.

This replaces the old two-pass flow (ffmpeg -> JPEG -> nvJPEG -> model) with a
single decode, so screenshots and the index together cost roughly one decode.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.io import encode_jpeg

from gpu_video_pipeline.decoder import NvdecSampler, frame_to_chw

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_for_screenshot(frame: torch.Tensor, max_width: int) -> torch.Tensor:
    """Match ffmpeg's ``scale='min(1280,iw)':-2`` (keep aspect, even height)."""
    _, height, width = frame.shape
    target_width = min(max_width, width)
    if target_width == width:
        return frame
    target_height = max(2, int(round(height * target_width / width)) // 2 * 2)
    resized = functional.interpolate(
        frame.unsqueeze(0).float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0).clamp_(0, 255).to(torch.uint8)


class CombinedIndexer:
    """Decode once; write screenshots and embeddings from the same frames."""

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        embedding_dim: int,
        screenshot_width: int = 1280,
        jpeg_quality: int = 90,
        device: str = "cuda",
    ) -> None:
        if image_size <= 0 or embedding_dim <= 0:
            raise ValueError("image_size and embedding_dim must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NVDEC indexing")
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.screenshot_width = screenshot_width
        self.jpeg_quality = jpeg_quality
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.scale = (1.0 / (255.0 * std)).to(torch.float16)
        self.bias = (-mean / std).to(torch.float16)

    def preprocess_images(self, frames: list[torch.Tensor]) -> torch.Tensor:
        images = torch.stack(frames)
        images = images.to(self.device, dtype=torch.float16)
        images = functional.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return images.mul_(self.scale).add_(self.bias)

    def index(
        self,
        sampler: NvdecSampler,
        output_dir: Path,
        screenshots_dir: Path,
        batch_size: int = 64,
    ) -> dict[str, object]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        partial = output_dir / "embeddings.npy.partial"
        completed = output_dir / "embeddings.npy"
        matrix = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.float16,
            shape=(len(sampler), self.embedding_dim),
        )
        records: list[dict[str, object]] = []
        position = 0
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for batch in sampler.batches(batch_size):
                    frames_chw = [frame_to_chw(frame) for frame in batch.frames]
                    shots = [
                        resize_for_screenshot(frame, self.screenshot_width)
                        for frame in frames_chw
                    ]
                    for offset, data in enumerate(
                        encode_jpeg(shots, quality=self.jpeg_quality)
                    ):
                        number = position + offset + 1
                        path = screenshots_dir / f"frame_{number:06d}.jpg"
                        path.write_bytes(data.cpu().numpy().tobytes())
                        records.append(
                            {
                                "row": number - 1,
                                "filename": path.name,
                                "frame_number": number,
                                "timestamp_seconds": batch.timestamps[offset],
                                "source_frame_index": batch.source_indices[offset],
                            }
                        )

                    images = self.preprocess_images(frames_chw)
                    with torch.autocast("cuda", dtype=torch.float16):
                        embeddings = self.model(images)
                    if embeddings.shape != (len(batch.frames), self.embedding_dim):
                        raise RuntimeError(
                            f"Model returned {tuple(embeddings.shape)}; expected "
                            f"({len(batch.frames)}, {self.embedding_dim})"
                        )
                    embeddings = functional.normalize(embeddings.float(), dim=1)
                    matrix[position : position + len(batch.frames)] = (
                        embeddings.to(torch.float16).cpu().numpy()
                    )
                    position += len(batch.frames)
            matrix.flush()
        except BaseException:
            del matrix
            partial.unlink(missing_ok=True)
            raise

        del matrix
        if position != len(sampler):
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Produced {position} of {len(sampler)} expected embeddings"
            )
        os.replace(partial, completed)
        (output_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        elapsed = time.perf_counter() - started
        manifest: dict[str, object] = {
            "source": str(sampler.movie),
            "source_size": sampler.movie.stat().st_size,
            "source_mtime_ns": sampler.movie.stat().st_mtime_ns,
            "decoder": "nvdec",
            "frames": position,
            "sample_fps": getattr(sampler, "sample_fps", None),
            "source_fps": sampler.source_fps,
            "duration_seconds": sampler.duration_seconds,
            "source_width": sampler.width,
            "source_height": sampler.height,
            "image_size": self.image_size,
            "embedding_dim": self.embedding_dim,
            "dtype": "float16",
            "batch_size": batch_size,
            "screenshot_width": self.screenshot_width,
            "jpeg_quality": self.jpeg_quality,
            "screenshots": str(screenshots_dir),
            "elapsed_seconds": elapsed,
            "frames_per_second": position / elapsed,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVDEC decode once; write JPEG screenshots and embeddings."
    )
    parser.add_argument("movie", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="TorchScript encoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--screenshot-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    model = torch.jit.load(str(args.model), map_location="cuda")
    sampler = NvdecSampler(args.movie, sample_fps=args.sample_fps)
    indexer = CombinedIndexer(
        model,
        args.image_size,
        args.embedding_dim,
        screenshot_width=args.screenshot_width,
        jpeg_quality=args.jpeg_quality,
    )
    manifest = indexer.index(
        sampler, args.output, args.screenshots, batch_size=args.batch_size
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
