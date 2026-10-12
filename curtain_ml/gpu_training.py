"""Combined global/local contrastive training for CUDA-compatible ROCm builds."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from curtain_ml.gpu_augmentations import GpuCombinedAugmenter
from curtain_ml.model import CurtainEncoder
from curtain_ml.training import (
    GridRetrievalDataset,
    RetrievalDataset,
    contrastive_loss,
    discover_movies,
    evaluate,
    seed_everything,
    select_paths,
    split_movies,
)


@dataclass(frozen=True)
class GpuTrainingConfig:
    data_dir: str
    output_dir: str
    epochs: int = 30
    batch_size: int = 256
    image_size: int = 192
    embedding_dim: int = 128
    learning_rate: float = 6e-4
    temperature: float = 0.07
    weight_decay: float = 1e-4
    workers: int = 20
    eval_movies: int = 6
    seed: int = 42
    limit: int | None = None
    view_mode: str = "combined_gpu"
    grid_size: int = 4
    tile_jitter: float = 0.5
    global_loss_weight: float = 0.5
    local_loss_weight: float = 0.5
    forward_chunk_size: int = 2048
    evaluation_interval: int = 2

    def __post_init__(self):
        if self.global_loss_weight < 0 or self.local_loss_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.global_loss_weight + self.local_loss_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if self.forward_chunk_size < 1:
            raise ValueError("Forward chunk size must be positive")
        if self.evaluation_interval < 1:
            raise ValueError("Evaluation interval must be positive")


class DecodedFrameDataset(Dataset):
    """Decode JPEGs on CPU; every learned transformation happens on the GPU."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return pil_to_tensor(image)


def _loaders(
    config: GpuTrainingConfig,
    train_paths: list[Path],
    eval_paths: list[Path],
) -> tuple[DataLoader, dict[str, DataLoader]]:
    # Variable-sized decoded images are returned as a list. Filesystem-backed
    # sharing avoids exhausting one file descriptor per tensor at large batch
    # sizes while workers pass those tensors to the training process.
    multiprocessing.set_sharing_strategy("file_system")
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    train_loader = DataLoader(
        DecodedFrameDataset(train_paths),
        shuffle=True,
        drop_last=True,
        collate_fn=list,
        # Full-resolution decoded frames are large. One queued batch per worker
        # keeps host/shared-memory use bounded on large logical GPU batches.
        prefetch_factor=1 if config.workers else None,
        **common,
    )
    evaluation_common = {
        **common,
        "batch_size": min(config.batch_size, 512),
    }
    return train_loader, {
        "global": DataLoader(
            RetrievalDataset(eval_paths, config.image_size),
            shuffle=False,
            drop_last=False,
            **evaluation_common,
        ),
        "grid": DataLoader(
            GridRetrievalDataset(
                eval_paths,
                config.image_size,
                config.seed + 100_000,
                config.grid_size,
                config.tile_jitter,
            ),
            shuffle=False,
            drop_last=False,
            **evaluation_common,
        ),
    }


def train(config: GpuTrainingConfig) -> dict[str, float]:
    """Train one encoder on global and 4x4-local objectives."""
    seed_everything(config.seed)
    data_dir = Path(config.data_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    movies = discover_movies(data_dir)
    train_movies, eval_movies = split_movies(movies, config.eval_movies, config.seed)
    train_paths = select_paths(movies, train_movies, config.limit, config.seed)
    eval_paths = select_paths(movies, eval_movies, config.limit, config.seed + 1)
    if len(train_paths) < config.batch_size or len(eval_paths) < 2:
        raise RuntimeError("The selected split is too small for this configuration.")

    split_manifest = {
        "training_movies": train_movies,
        "evaluation_movies": eval_movies,
        "training_frames": len(train_paths),
        "evaluation_frames": len(eval_paths),
    }
    (output_dir / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Combined training requires a CUDA or ROCm PyTorch build.")

    device = torch.device("cuda")
    train_loader, evaluation_loaders = _loaders(config, train_paths, eval_paths)
    augmenter = GpuCombinedAugmenter(config.image_size, config.grid_size)
    model = CurtainEncoder(config.embedding_dim, config.image_size).to(device)
    model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    warmup_epochs = max(1, min(3, config.epochs // 5))

    def learning_rate_multiplier(scheduler_epoch: int) -> float:
        completed_epoch = scheduler_epoch + 1
        if completed_epoch <= warmup_epochs:
            return completed_epoch / warmup_epochs
        progress = (completed_epoch - warmup_epochs) / max(
            1, config.epochs - warmup_epochs
        )
        return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    scaler = torch.amp.GradScaler("cuda")
    best_score = -1.0
    final_metrics: dict[str, float] = {}
    print(
        json.dumps(
            {
                **split_manifest,
                "device": str(device),
                "rocm": torch.version.hip,
                "gpu": torch.cuda.get_device_name(0),
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats()
        total_loss = total_global = total_local = 0.0
        for images in train_loader:
            optimizer.zero_grad(set_to_none=True)
            full, global_crop, exact_tile, shifted_tile = augmenter(images, device)
            # Interleave view families so BatchNorm sees the same distribution
            # in every forward chunk. Capping the physical forward batch also
            # avoids MIOpen's very-large-N BatchNorm kernel limit while the
            # contrastive loss still uses the complete logical source batch.
            views = torch.stack(
                (full, global_crop, exact_tile, shifted_tile), dim=1
            ).flatten(0, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                embeddings = torch.cat(
                    [
                        model(views[offset : offset + config.forward_chunk_size])
                        for offset in range(0, len(views), config.forward_chunk_size)
                    ]
                ).reshape(len(images), 4, -1)
                (
                    full_embedding,
                    global_embedding,
                    exact_embedding,
                    shifted_embedding,
                ) = embeddings.unbind(dim=1)
                global_loss = contrastive_loss(
                    full_embedding, global_embedding, config.temperature
                )
                local_loss = contrastive_loss(
                    exact_embedding, shifted_embedding, config.temperature
                )
                loss = (
                    config.global_loss_weight * global_loss
                    + config.local_loss_weight * local_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_global += global_loss.item()
            total_local += local_loss.item()
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        scheduler.step()

        steps = max(1, len(train_loader))
        training_metrics = {
            "epoch": epoch,
            "training_loss": total_loss / steps,
            "global_loss": total_global / steps,
            "local_loss": total_local / steps,
            "learning_rate": epoch_learning_rate,
            "training_seconds": training_seconds,
            "source_frames_per_second": steps
            * config.batch_size
            / training_seconds,
            "model_views_per_second": steps
            * config.batch_size
            * 4
            / training_seconds,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        should_evaluate = (
            epoch == 1
            or epoch % config.evaluation_interval == 0
            or epoch == config.epochs
        )
        if should_evaluate:
            global_metrics = evaluate(model, evaluation_loaders["global"], device)
            grid_metrics = evaluate(model, evaluation_loaders["grid"], device)
            selection_score = 0.5 * (
                global_metrics["exact_frame_top1"]
                + grid_metrics["exact_frame_top1"]
            )
            final_metrics = {
                **training_metrics,
                "global_exact_frame_top1": global_metrics["exact_frame_top1"],
                "global_movie_top1": global_metrics["movie_top1"],
                "grid_exact_frame_top1": grid_metrics["exact_frame_top1"],
                "grid_movie_top1": grid_metrics["movie_top1"],
                "selection_score": selection_score,
                "evaluated": True,
                "epoch_seconds": time.perf_counter() - started,
            }
            checkpoint = {
                "model": model.state_dict(),
                "config": asdict(config),
                "metrics": final_metrics,
            }
            if selection_score > best_score:
                best_score = selection_score
                torch.save(checkpoint, output_dir / "best.pt")
        else:
            final_metrics = {
                **training_metrics,
                "evaluated": False,
                "epoch_seconds": time.perf_counter() - started,
            }
        print(json.dumps(final_metrics), flush=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n"
    )
    return final_metrics


def parse_args() -> GpuTrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/selected_screenshots"))
    parser.add_argument("--output", type=Path, default=Path("training_runs/combined"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--forward-chunk-size", type=int, default=2048)
    parser.add_argument("--evaluation-interval", type=int, default=2)
    args = parser.parse_args()
    return GpuTrainingConfig(
        data_dir=str(args.data),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        limit=args.limit,
        forward_chunk_size=args.forward_chunk_size,
        evaluation_interval=args.evaluation_interval,
    )


if __name__ == "__main__":
    train(parse_args())
