from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from curtain_ml.augmentations import RandomFrameCrop, RandomLighting
from curtain_ml.model import CurtainEncoder


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters shared by local and Modal contrastive-training runs."""
    data_dir: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 64
    image_size: int = 192
    embedding_dim: int = 128
    learning_rate: float = 3e-4
    temperature: float = 0.07
    weight_decay: float = 1e-4
    workers: int = 8
    eval_movies: int = 6
    seed: int = 42
    limit: int | None = None


class RandomJpegCompression:
    """Round-trip a PIL image through JPEG at a random quality."""

    def __init__(
        self,
        minimum_quality: int = 45,
        maximum_quality: int = 95,
        probability: float = 0.7,
    ):
        # Configure a probabilistic JPEG round-trip for one PIL image.
        self.minimum_quality = minimum_quality
        self.maximum_quality = maximum_quality
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        # Return the original image or a randomly quality-compressed copy.
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(self.minimum_quality, self.maximum_quality),
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class RandomResolutionDegradation:
    """Simulate an online image that was reduced and then enlarged again."""

    def __init__(self, minimum_scale: float = 0.45, probability: float = 0.5):
        # Configure the probability and smallest temporary resolution.
        self.minimum_scale = minimum_scale
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        # Downsample and upsample an image to simulate a lower-resolution source.
        if random.random() >= self.probability:
            return image
        width, height = image.size
        scale = random.uniform(self.minimum_scale, 0.9)
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=random.choice(
                (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
            ),
        )
        return reduced.resize(
            (width, height),
            resample=random.choice(
                (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
            ),
        )


def appearance_transform(image_size: int) -> transforms.Compose:
    # Lighting and compression changes that preserve a frame's identity.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            RandomLighting(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2
            ),
            RandomResolutionDegradation(),
            RandomJpegCompression(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def training_transform(image_size: int) -> transforms.Compose:
    # The active crop-v1 positive-pair transformation.
    return transforms.Compose([RandomFrameCrop(), appearance_transform(image_size)])


def reference_transform(image_size: int) -> transforms.Compose:
    # Build the deterministic preprocessing used for indexed reference frames.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def discover_movies(data_dir: Path) -> dict[str, list[Path]]:
    # Find populated movie directories and their supported image files.
    movies: dict[str, list[Path]] = {}
    for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        frames = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if frames:
            movies[directory.name] = frames
    if len(movies) < 2:
        raise RuntimeError(
            f"Expected at least two populated movie folders in {data_dir}."
        )
    return movies


def split_movies(
    movies: dict[str, list[Path]], eval_movies: int, seed: int
) -> tuple[list[str], list[str]]:
    # Split complete movie titles into deterministic train and evaluation sets.
    names = sorted(movies)
    random.Random(seed).shuffle(names)
    if not 1 <= eval_movies < len(names):
        raise ValueError("eval_movies must leave at least one movie in each split.")
    return sorted(names[eval_movies:]), sorted(names[:eval_movies])


def select_paths(
    movies: dict[str, list[Path]], names: list[str], limit: int | None, seed: int
) -> list[Path]:
    # Flatten selected movies and optionally sample a reproducible frame limit.
    paths = [path for name in names for path in movies[name]]
    if limit is not None and limit < len(paths):
        paths = random.Random(seed).sample(paths, limit)
    return sorted(paths)


class PositivePairDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        # Prepare paired transforms for the same source frame.
        self.paths = paths
        self.anchor_transform = appearance_transform(image_size)
        self.positive_transform = training_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load one frame and return its anchor and augmented positive views.
        with Image.open(self.paths[index]) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        return self.anchor_transform(source), self.positive_transform(source)


class RetrievalDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        # Prepare deterministic references and appearance-varied query views.
        self.paths = paths
        self.reference = reference_transform(image_size)
        self.query = appearance_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        # Return reference/query tensors plus the source index and movie name.
        path = self.paths[index]
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        return self.reference(source), self.query(source), index, path.parent.name


def contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float
) -> torch.Tensor:
    # Compute symmetric in-batch InfoNCE for two views of every source frame.
    batch_size = first.shape[0]
    embeddings = torch.cat((first, second), dim=0)
    logits = embeddings.float() @ embeddings.float().T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * batch_size, device=embeddings.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return functional.cross_entropy(logits, targets)


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    # Measure exact-frame and movie-level top-1 retrieval on one split.
    model.eval()
    references: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    movies: list[str] = []
    for reference, query, index, movie in loader:
        references.append(model(reference.to(device)).cpu())
        queries.append(model(query.to(device)).cpu())
        indices.append(index)
        movies.extend(movie)

    reference_embeddings = torch.cat(references)
    query_embeddings = torch.cat(queries)
    expected = torch.cat(indices)
    predicted = (query_embeddings @ reference_embeddings.T).argmax(dim=1)
    exact = (predicted == expected).float().mean().item()
    movie_matches = [
        movies[prediction] == movies[target]
        for prediction, target in zip(predicted.tolist(), expected.tolist())
    ]
    return {
        "exact_frame_top1": exact,
        "movie_top1": sum(movie_matches) / len(movie_matches),
    }


def seed_everything(seed: int) -> None:
    # Seed Python, Torch, and all visible CUDA devices.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: TrainingConfig) -> dict[str, float]:
    # Train the encoder on Modal's CUDA device and save the best checkpoint.
    seed_everything(config.seed)
    data_dir = Path(config.data_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    movies = discover_movies(data_dir)
    train_movies, eval_movies = split_movies(movies, config.eval_movies, config.seed)
    train_paths = select_paths(movies, train_movies, config.limit, config.seed)
    eval_paths = select_paths(movies, eval_movies, config.limit, config.seed + 1)
    if len(train_paths) < 2 or len(eval_paths) < 2:
        raise RuntimeError("Both splits need at least two frames.")

    split_manifest = {
        "training_movies": train_movies,
        "evaluation_movies": eval_movies,
        "training_frames": len(train_paths),
        "evaluation_frames": len(eval_paths),
    }
    (output_dir / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError("Curtain training requires CUDA.")
    device = torch.device("cuda")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    train_loader = DataLoader(
        PositivePairDataset(train_paths, config.image_size),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        RetrievalDataset(eval_paths, config.image_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = CurtainEncoder(config.embedding_dim, config.image_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    best_accuracy = -1.0
    final_metrics: dict[str, float] = {}

    print(json.dumps({**split_manifest, "device": str(device)}, indent=2), flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        for first, second in train_loader:
            optimizer.zero_grad(set_to_none=True)
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                first_embeddings = model(first)
                second_embeddings = model(second)
                loss = contrastive_loss(
                    first_embeddings, second_embeddings, config.temperature
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        final_metrics = evaluate(model, eval_loader, device)
        final_metrics.update(
            epoch=epoch,
            training_loss=total_loss / max(1, len(train_loader)),
            epoch_seconds=time.perf_counter() - started,
        )
        print(json.dumps(final_metrics), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "metrics": final_metrics,
        }
        if final_metrics["exact_frame_top1"] > best_accuracy:
            best_accuracy = final_metrics["exact_frame_top1"]
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    return final_metrics


def parse_args() -> TrainingConfig:
    # Parse the intentionally small local training command line.
    parser = argparse.ArgumentParser(
        description="Train Curtain's contrastive frame encoder."
    )
    parser.add_argument("--data", type=Path, default=Path("selected_screenshots"))
    parser.add_argument("--output", type=Path, default=Path("training_runs/default"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    return TrainingConfig(
        data_dir=str(args.data),
        output_dir=str(args.output),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    train(parse_args())
