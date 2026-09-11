"""Stream decoded movie batches through a PyTorch encoder."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as functional

from gpu_video_pipeline.decoder import NvdecSampler, SampledBatch, frame_to_chw


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Sampler(Protocol):
    movie: Path
    duration_seconds: float
    source_fps: float
    width: int
    height: int

    def __len__(self) -> int: ...

    def batches(self, batch_size: int): ...


class VideoIndexer:
    """Generic model runner; it has no dependency on Curtain."""

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        embedding_dim: int,
        device: str = "cuda",
        use_float16: bool = True,
    ):
        if image_size <= 0 or embedding_dim <= 0:
            raise ValueError("image_size and embedding_dim must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NVDEC indexing")
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.use_float16 = use_float16 and self.device.type == "cuda"
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, batch: SampledBatch) -> torch.Tensor:
        images = torch.stack([frame_to_chw(frame) for frame in batch.frames])
        images = images.to(self.device, dtype=torch.float32).div_(255.0)
        images = functional.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return images.sub_(self.mean).div_(self.std)

    def index(
        self,
        sampler: Sampler,
        output_dir: Path,
        batch_size: int = 64,
    ) -> dict[str, object]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = output_dir / "embeddings.npy.partial"
        completed = output_dir / "embeddings.npy"
        matrix = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.float16,
            shape=(len(sampler), self.embedding_dim),
        )
        records = []
        position = 0
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for batch in sampler.batches(batch_size):
                    images = self.preprocess(batch)
                    autocast = (
                        torch.autocast("cuda", dtype=torch.float16)
                        if self.use_float16
                        else nullcontext()
                    )
                    with autocast:
                        embeddings = self.model(images)
                    if embeddings.shape != (len(batch.frames), self.embedding_dim):
                        raise RuntimeError(
                            f"Model returned {tuple(embeddings.shape)}; expected "
                            f"({len(batch.frames)}, {self.embedding_dim})"
                        )
                    embeddings = functional.normalize(embeddings.float(), dim=1)
                    count = len(batch.frames)
                    matrix[position : position + count] = (
                        embeddings.to(torch.float16).cpu().numpy()
                    )
                    records.extend(
                        {
                            "row": position + offset,
                            "timestamp_seconds": timestamp,
                            "source_frame_index": source_index,
                        }
                        for offset, (timestamp, source_index) in enumerate(
                            zip(batch.timestamps, batch.source_indices, strict=True)
                        )
                    )
                    position += count
            matrix.flush()
        except BaseException:
            del matrix
            partial.unlink(missing_ok=True)
            raise

        del matrix
        if position != len(sampler):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Produced {position} of {len(sampler)} expected embeddings")
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
            "elapsed_seconds": elapsed,
            "frames_per_second": position / elapsed,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def index_movie(
    movie: Path,
    model: torch.nn.Module,
    output_dir: Path,
    image_size: int,
    embedding_dim: int,
    sample_fps: float = 2.0,
    batch_size: int = 64,
) -> dict[str, object]:
    sampler = NvdecSampler(movie, sample_fps=sample_fps)
    indexer = VideoIndexer(model, image_size, embedding_dim)
    return indexer.index(sampler, output_dir, batch_size=batch_size)
