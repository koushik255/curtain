from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/sscd_disc_mixup.torchscript.pt"
DEFAULT_QUANTIZED_MODEL = (
    PROJECT_ROOT / "models/sscd_disc_mixup.int8-experiment.torchscript.pt"
)
DEFAULT_TESTING = PROJECT_ROOT / "testing"
DEFAULT_INDEX = PROJECT_ROOT / "index"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((320, 320), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    batch = torch.from_numpy(np.transpose(pixels, (2, 0, 1))).unsqueeze(0)
    return (batch - MEAN) / STD


def split_images(directory: Path) -> tuple[list[Path], list[Path]]:
    by_movie: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            movie_id = path.stem.rsplit("__frame_", 1)[0]
            by_movie[movie_id].append(path)

    calibration = []
    evaluation = []
    for paths in by_movie.values():
        if len(paths) < 2:
            continue
        calibration.append(paths[0])
        evaluation.append(paths[-1])
    if not evaluation:
        raise RuntimeError("Testing requires at least two named images per movie.")
    return calibration, evaluation


def quantize_model(
    model: torch.jit.ScriptModule,
    calibration_batches: list[torch.Tensor],
    engine: str,
    qconfig_backend: str,
    quantize_head: bool,
) -> torch.jit.ScriptModule:
    torch.backends.quantized.engine = engine
    qconfig = torch.ao.quantization.get_default_qconfig(qconfig_backend)

    def calibrate(candidate, batches):
        with torch.inference_mode():
            for batch in batches:
                candidate(batch)

    qconfig_dict = {"": qconfig}
    if not quantize_head:
        qconfig_dict["embeddings"] = None

    with torch.inference_mode():
        return torch.ao.quantization.quantize_jit(
            model,
            qconfig_dict,
            calibrate,
            [calibration_batches],
            inplace=False,
        )


def encode(model: torch.jit.ScriptModule, batch: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        values = model(batch).float().cpu().numpy()[0]
    return values / max(float(np.linalg.norm(values)), 1e-12)


def encode_and_time(
    model: torch.jit.ScriptModule,
    batches: list[torch.Tensor],
    warmups: int,
) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmups):
        encode(model, batches[0])

    embeddings = []
    timings = []
    for batch in batches:
        started = time.perf_counter()
        embeddings.append(encode(model, batch))
        timings.append((time.perf_counter() - started) * 1000)
    return np.stack(embeddings), timings


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
    }


def search_all(
    index_dir: Path, queries: np.ndarray
) -> tuple[list[str], list[tuple[int, int]]]:
    best_scores = np.full(len(queries), -np.inf, dtype=np.float32)
    best_movies = [""] * len(queries)
    best_locations = [(-1, -1)] * len(queries)

    manifest_paths = sorted((index_dir / "movies").glob("*/manifest.json"))
    if not manifest_paths:
        raise RuntimeError(f"No completed index shards found in {index_dir}.")

    query_columns = queries.T
    for shard_index, manifest_path in enumerate(manifest_paths):
        manifest = json.loads(manifest_path.read_text())
        stored = np.load(manifest_path.parent / "embeddings.npy", mmap_mode="r")
        searchable = np.asarray(stored, dtype=np.float32)
        scores = searchable @ query_columns
        local_indices = np.argmax(scores, axis=0)
        local_scores = scores[local_indices, np.arange(len(queries))]
        for query_index, score in enumerate(local_scores):
            if score > best_scores[query_index]:
                best_scores[query_index] = score
                best_movies[query_index] = manifest["movie_id"]
                best_locations[query_index] = (
                    shard_index,
                    int(local_indices[query_index]),
                )
    return best_movies, best_locations


def mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small FP32-versus-INT8 experiment for Curtain's SSCD model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_QUANTIZED_MODEL)
    parser.add_argument("--testing", type=Path, default=DEFAULT_TESTING)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--engine", choices=("x86", "fbgemm", "onednn"), default="onednn"
    )
    parser.add_argument(
        "--qconfig-backend",
        choices=("x86", "fbgemm", "onednn"),
        help="Quantization calibration profile (defaults to --engine).",
    )
    parser.add_argument(
        "--quantize-head",
        action="store_true",
        help="Also quantize the final descriptor projection (FP32 by default).",
    )
    parser.add_argument("--warmups", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_paths, evaluation_paths = split_images(args.testing)
    print(
        f"Using {len(calibration_paths)} calibration images and "
        f"{len(evaluation_paths)} held-out evaluation images."
    )
    calibration_batches = [prepare_image(path) for path in calibration_paths]
    evaluation_batches = [prepare_image(path) for path in evaluation_paths]

    fp32_model = torch.jit.load(str(args.model), map_location="cpu").eval()
    qconfig_backend = args.qconfig_backend or args.engine
    started = time.perf_counter()
    int8_model = quantize_model(
        fp32_model,
        calibration_batches,
        args.engine,
        qconfig_backend,
        args.quantize_head,
    ).eval()
    quantization_seconds = time.perf_counter() - started
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(int8_model, str(args.output_model))

    fp32_embeddings, fp32_times = encode_and_time(
        fp32_model, evaluation_batches, args.warmups
    )
    int8_embeddings, int8_times = encode_and_time(
        int8_model, evaluation_batches, args.warmups
    )
    descriptor_cosines = np.sum(fp32_embeddings * int8_embeddings, axis=1)

    combined = np.concatenate((fp32_embeddings, int8_embeddings), axis=0)
    predicted_movies, locations = search_all(args.index, combined)
    split = len(evaluation_paths)
    fp32_movies, int8_movies = predicted_movies[:split], predicted_movies[split:]
    fp32_locations, int8_locations = locations[:split], locations[split:]
    expected_movies = [path.stem.rsplit("__frame_", 1)[0] for path in evaluation_paths]

    fp32_correct = sum(a == b for a, b in zip(fp32_movies, expected_movies))
    int8_correct = sum(a == b for a, b in zip(int8_movies, expected_movies))
    movie_agreement = sum(a == b for a, b in zip(fp32_movies, int8_movies))
    frame_agreement = sum(a == b for a, b in zip(fp32_locations, int8_locations))
    count = len(evaluation_paths)
    fp32_latency = latency_summary(fp32_times)
    int8_latency = latency_summary(int8_times)

    results = {
        "calibration_images": len(calibration_paths),
        "evaluation_images": count,
        "quantization_seconds": quantization_seconds,
        "engine": args.engine,
        "qconfig_backend": qconfig_backend,
        "quantized_head": args.quantize_head,
        "model_size_mib": {"fp32": mib(args.model), "int8": mib(args.output_model)},
        "forward_latency": {"fp32": fp32_latency, "int8": int8_latency},
        "median_speedup": fp32_latency["median_ms"] / int8_latency["median_ms"],
        "descriptor_cosine": {
            "mean": float(np.mean(descriptor_cosines)),
            "minimum": float(np.min(descriptor_cosines)),
        },
        "top1_movie_accuracy": {
            "fp32": fp32_correct / count,
            "int8": int8_correct / count,
        },
        "fp32_int8_movie_agreement": movie_agreement / count,
        "fp32_int8_exact_frame_agreement": frame_agreement / count,
        "movie_mismatches": [
            {
                "image": path.name,
                "expected": expected,
                "fp32": fp32,
                "int8": int8,
            }
            for path, expected, fp32, int8 in zip(
                evaluation_paths, expected_movies, fp32_movies, int8_movies
            )
            if fp32 != int8 or int8 != expected
        ],
        "output_model": str(args.output_model),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
