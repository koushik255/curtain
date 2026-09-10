from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from curtain_ml.training import (
    CurtainEncoder,
    discover_movies,
    reference_transform,
    split_movies,
    training_transform,
)


class FixedRetrievalDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, query_seed: int | None):
        self.paths = paths
        self.reference = reference_transform(image_size)
        self.query = training_transform(image_size)
        self.query_seed = query_seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path = self.paths[index]
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        if self.query_seed is None:
            transformed = self.reference(source)
        else:
            # A seed derived from the frame index makes each changed query
            # identical across models, workers, and repeated benchmark runs.
            python_state = random.getstate()
            torch_state = torch.random.get_rng_state()
            random.seed(self.query_seed + index)
            torch.manual_seed(self.query_seed + index)
            transformed = self.query(source)
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
        return transformed, index, 0


@torch.inference_mode()
def encode(
    model: CurtainEncoder,
    paths: list[Path],
    image_size: int,
    query_seed: int | None,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        FixedRetrievalDataset(paths, image_size, query_seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    outputs = []
    for images, _indices, _unused in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs.append(model(images.to(device, non_blocking=True)).float().cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def retrieval_metrics(
    references: torch.Tensor,
    queries: torch.Tensor,
    movie_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int = 1024,
) -> dict[str, float]:
    references = references.to(device)
    exact = 0
    movie = 0
    reciprocal_rank = 0.0
    for offset in range(0, len(queries), chunk_size):
        end = min(offset + chunk_size, len(queries))
        scores = queries[offset:end].to(device) @ references.T
        expected = torch.arange(offset, end, device=device)
        predicted = scores.argmax(dim=1)
        exact += (predicted == expected).sum().item()
        movie += (movie_ids[predicted.cpu()] == movie_ids[offset:end]).sum().item()
        target_scores = scores[torch.arange(end - offset, device=device), expected]
        ranks = (scores > target_scores[:, None]).sum(dim=1) + 1
        reciprocal_rank += (1.0 / ranks.float()).sum().item()
    count = len(queries)
    return {
        "exact_frame_top1": exact / count,
        "movie_top1": movie / count,
        "mean_reciprocal_rank": reciprocal_rank / count,
    }


def compare(
    data_dir: Path,
    checkpoints: dict[str, Path],
    output_path: Path,
    seeds: tuple[int, ...] = (10_001, 20_002, 30_003),
    batch_size: int = 256,
    workers: int = 8,
    device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name)
    movies = discover_movies(data_dir)
    _train_movies, eval_movies = split_movies(movies, eval_movies=6, seed=42)
    paths = sorted(path for name in eval_movies for path in movies[name])
    movie_lookup = {name: index for index, name in enumerate(eval_movies)}
    movie_ids = torch.tensor([movie_lookup[path.parent.name] for path in paths])
    result = {
        "frames": len(paths),
        "evaluation_movies": eval_movies,
        "augmentation_seeds": list(seeds),
        "models": {},
    }

    for name, checkpoint_path in checkpoints.items():
        started = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        image_size = int(config["image_size"])
        model = CurtainEncoder(int(config["embedding_dim"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        references = encode(model, paths, image_size, None, device, batch_size, workers)
        passes = []
        for seed in seeds:
            queries = encode(model, paths, image_size, seed, device, batch_size, workers)
            metrics = retrieval_metrics(references, queries, movie_ids, device)
            metrics["seed"] = seed
            passes.append(metrics)
        averages = {
            key: sum(item[key] for item in passes) / len(passes)
            for key in ("exact_frame_top1", "movie_top1", "mean_reciprocal_rank")
        }
        result["models"][name] = {
            "checkpoint": checkpoint_path.name,
            "passes": passes,
            "average": averages,
            "benchmark_seconds": time.perf_counter() - started,
        }
        print(json.dumps({name: result["models"][name]}, indent=2), flush=True)

    old = result["models"]["5k_baseline"]["average"]
    new = result["models"]["50k_model"]["average"]
    result["difference_50k_minus_5k"] = {
        key: new[key] - old[key] for key in old
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"difference_50k_minus_5k": result["difference_50k_minus_5k"]}, indent=2), flush=True)
    return result
