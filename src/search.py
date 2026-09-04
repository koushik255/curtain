from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.model import DEFAULT_MODEL_PATH, PROJECT_ROOT, SSCDModel


DEFAULT_INDEX = PROJECT_ROOT / "index"


class SearchEngine:
    """Keep SSCD and every completed per-movie index loaded for repeated searches."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX,
        model_path: Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        preload: bool = False,
    ) -> None:
        self.index_dir = index_dir.expanduser().resolve()
        manifest_paths = sorted((self.index_dir / "movies").glob("*/manifest.json"))
        if not manifest_paths:
            raise RuntimeError("The index is missing. Run `uv run python -m src.index` first.")

        self.shards: list[dict] = []
        for manifest_path in manifest_paths:
            directory = manifest_path.parent
            manifest = json.loads(manifest_path.read_text())
            embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
            records = json.loads((directory / "records.json").read_text())
            if embeddings.shape != (len(records), SSCDModel.dimensions):
                raise RuntimeError(f"Index files disagree in {directory}.")
            if manifest.get("dtype") and manifest["dtype"] != embeddings.dtype.name:
                raise RuntimeError(f"Index dtype disagrees in {directory}.")
            if preload:
                embeddings = np.asarray(embeddings, dtype=np.float32)
            self.shards.append(
                {"manifest": manifest, "embeddings": embeddings, "records": records}
            )

        self.model = SSCDModel(model_path, device)
        self.total_frames = sum(len(shard["records"]) for shard in self.shards)
        print(f"Loaded {len(self.shards)} movies and {self.total_frames:,} frame embeddings.")

    @staticmethod
    def resolve_image_path(record: dict) -> Path:
        path = Path(record["image_path"])
        return path if path.is_absolute() else PROJECT_ROOT / path

    def record(self, shard_index: int, embedding_index: int) -> dict | None:
        if not 0 <= shard_index < len(self.shards):
            return None
        records = self.shards[shard_index]["records"]
        if not 0 <= embedding_index < len(records):
            return None
        return records[embedding_index]

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        timings: dict | None = None,
    ) -> list[dict]:
        started = time.perf_counter()
        query = self.model.encode([image.convert("RGB")])[0]
        embedded = time.perf_counter()

        # Each shard is one movie. Its highest-scoring frame becomes that movie's score.
        candidates = []
        for shard_index, shard in enumerate(self.shards):
            # Stored float16 descriptors are promoted so dot products accumulate
            # in float32. This preserves ranking accuracy while halving index size.
            searchable = np.asarray(shard["embeddings"], dtype=np.float32)
            scores = searchable @ query
            embedding_index = int(np.argmax(scores))
            candidates.append(
                {
                    **shard["records"][embedding_index],
                    # Quantization can move a cosine score a few millionths
                    # beyond its mathematical [-1, 1] range.
                    "score": float(np.clip(scores[embedding_index], -1.0, 1.0)),
                    "shard_index": shard_index,
                    "embedding_index": embedding_index,
                }
            )
        compared = time.perf_counter()
        candidates.sort(key=lambda item: item["score"], reverse=True)
        results = candidates[: max(1, min(top_k, len(candidates)))]
        ranked = time.perf_counter()

        if timings is not None:
            timings.update(
                embedding_ms=(embedded - started) * 1000,
                comparison_ms=(compared - embedded) * 1000,
                ranking_ms=(ranked - compared) * 1000,
            )
        return results

    def search_bytes(
        self,
        data: bytes,
        top_k: int = 5,
        timings: dict | None = None,
    ) -> list[dict]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            decoded = time.perf_counter()
            results = self.search_image(image, top_k, timings)
        if timings is not None:
            timings["decode_ms"] = (decoded - started) * 1000
            timings["search_total_ms"] = (time.perf_counter() - started) * 1000
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a screenshot's movie with SSCD.")
    parser.add_argument("query", type=Path)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    whole = round(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    args = parse_args()
    engine = SearchEngine(args.index, args.model, args.device)
    with Image.open(args.query) as image:
        results = engine.search_image(image, args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for rank, result in enumerate(results, start=1):
        year = f" ({result['year']})" if result["year"] else ""
        print(
            f"{rank}. {result['title']}{year} | "
            f"{format_timestamp(result['timestamp_seconds'])} | score {result['score']:.4f}"
        )
        print(f"   {result['image_path']}")


if __name__ == "__main__":
    main()
