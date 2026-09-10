"""Index every movie on two parallel Modal L4 workers."""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "curtain-training-data"
RUN = "l4-50k-crop-v1-20260909"
CHECKPOINT_HASH = "a35b1528a734e451f1165418acf4ca24b5a5cc27d5c6be7dab461663c640ab3d"
CHECKPOINT = Path(f"/data/runs/{RUN}/best.pt")
COLLECTIONS = (
    ("l4-50k", "/data/datasets/full-curtain", "l4-50k-crop-v1"),
    ("l4-50k-lbfive", "/data/datasets/l4-50k-lbfive", "l4-50k-lbfive-crop-v1"),
)

app = modal.App("curtain-index")
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=2.2,<3", "torch>=2.8,<3", "torchvision>=0.23,<1")
    .add_local_dir("curtain_ml", remote_path="/root/curtain_ml", copy=True)
)


@app.cls(
    image=image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=60 * 60,
    volumes={"/data": volume},
    max_containers=2,
    scaledown_window=10 * 60,
)
class IndexWorker:
    @modal.enter()
    def start(self) -> None:
        # Load and verify the shared checkpoint once per L4 container.
        from curtain_ml.index import CudaIndexer

        volume.reload()
        actual = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        if actual != CHECKPOINT_HASH:
            raise RuntimeError(f"Unexpected checkpoint hash: {actual}")
        self.indexer = CudaIndexer(CHECKPOINT)

    @modal.method()
    def index(self, task: dict) -> dict:
        # Extract, encode, validate, and commit one movie's index.
        import numpy as np

        output = Path("/data/indexes") / task["output_collection"] / task["movie"]
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            matrix = np.load(output / "embeddings.npy", mmap_mode="r")
            valid = (
                manifest.get("checkpoint_sha256") == CHECKPOINT_HASH
                and manifest.get("source_sha256") == task["source_sha256"]
                and manifest.get("frames") == task["frames"]
                and manifest.get("decoder") == "nvjpeg"
                and matrix.shape == (task["frames"], 128)
                and (output / "records.json").is_file()
            )
            if valid:
                return {**manifest, "skipped": True}
            raise RuntimeError(f"Existing index does not match task: {output}")

        started = time.perf_counter()
        archive_path = Path(task["archive_dir"]) / f'{task["movie"]}.tar'
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)

        with tempfile.TemporaryDirectory(prefix="curtain-index-") as temporary:
            movie_dir = Path(temporary) / task["movie"]
            movie_dir.mkdir()
            with tarfile.open(archive_path) as archive:
                archive.extractall(movie_dir, filter="data")
            manifest = self.indexer.index_movie(movie_dir, output)

        if manifest["frames"] != task["frames"]:
            raise RuntimeError(
                f'{task["movie"]}: expected {task["frames"]} frames, '
                f'found {manifest["frames"]}'
            )
        matrix = np.load(output / "embeddings.npy", mmap_mode="r")
        if matrix.shape != (task["frames"], 128) or not np.isfinite(matrix).all():
            raise RuntimeError(f'Invalid embeddings for {task["movie"]}')

        manifest.update(
            checkpoint_sha256=CHECKPOINT_HASH,
            checkpoint_run=RUN,
            source_sha256=task["source_sha256"],
            source_root=task["source_root"],
            source_collection=task["source_collection"],
            source_archive=str(archive_path),
            output_collection=task["output_collection"],
            sample_fps=2,
            worker_seconds=time.perf_counter() - started,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        print(
            f'COMPLETE {task["output_collection"]}/{task["movie"]}: '
            f'{manifest["frames"]:,} frames',
            flush=True,
        )
        return {**manifest, "skipped": False}


def tasks() -> list[dict]:
    # Build one independent indexing task for every source movie.
    result = []
    for source_collection, archive_dir, output_collection in COLLECTIONS:
        source = ROOT / "trained_indexes" / source_collection
        collection = json.loads((source / "collection.json").read_text())
        for movie in collection["movies"]:
            manifest = json.loads((source / movie["movie"] / "manifest.json").read_text())
            result.append(
                {
                    "source_collection": source_collection,
                    "archive_dir": archive_dir,
                    "output_collection": output_collection,
                    "movie": movie["movie"],
                    "frames": int(movie["frames"]),
                    "source_sha256": manifest["source_sha256"],
                    "source_root": manifest["source_root"],
                }
            )
    return result


def download_and_verify(all_tasks: list[dict]) -> None:
    # Download completed Modal indexes and verify their provenance and shape.
    import numpy as np

    expected = {(task["output_collection"], task["movie"]): task for task in all_tasks}
    for source_collection, _archive_dir, output_collection in COLLECTIONS:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                "--force",
                VOLUME_NAME,
                f"/indexes/{output_collection}",
                str(ROOT / "trained_indexes"),
            ],
            check=True,
        )
        destination = ROOT / "trained_indexes" / output_collection
        source = ROOT / "trained_indexes" / source_collection
        source_manifest = json.loads((source / "collection.json").read_text())
        manifests = []
        for movie in source_manifest["movies"]:
            name = movie["movie"]
            task = expected[(output_collection, name)]
            folder = destination / name
            manifest = json.loads((folder / "manifest.json").read_text())
            records = json.loads((folder / "records.json").read_text())
            matrix = np.load(folder / "embeddings.npy", mmap_mode="r")
            if (
                manifest.get("checkpoint_sha256") != CHECKPOINT_HASH
                or manifest.get("source_sha256") != task["source_sha256"]
                or len(records) != task["frames"]
                or matrix.shape != (task["frames"], 128)
                or not np.isfinite(matrix).all()
            ):
                raise RuntimeError(f"Verification failed: {output_collection}/{name}")
            manifests.append(manifest)

        collection = {
            "checkpoint_sha256": CHECKPOINT_HASH,
            "checkpoint_run": RUN,
            "source_collection": source_collection,
            "movies": manifests,
            "total_frames": sum(item["frames"] for item in manifests),
        }
        (destination / "collection.json").write_text(json.dumps(collection, indent=2) + "\n")
        print(
            f'VERIFIED {output_collection}: {len(manifests)} movies, '
            f'{collection["total_frames"]:,} frames',
            flush=True,
        )


@app.local_entrypoint()
def main(limit_movies: int = 0, download_results: bool = True, dry_run: bool = False) -> None:
    # Plan or run the two-worker indexing job from the local machine.
    all_tasks = tasks()
    selected = all_tasks[:limit_movies] if limit_movies else all_tasks
    print(
        f"Prepared {len(selected)} movies and "
        f'{sum(task["frames"] for task in selected):,} frames for two L4 workers.',
        flush=True,
    )
    if dry_run:
        return
    list(IndexWorker().index.map(selected, order_outputs=False))
    if download_results and not limit_movies:
        download_and_verify(all_tasks)
