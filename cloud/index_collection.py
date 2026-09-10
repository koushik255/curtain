"""Resumable full-resolution archive upload and L4 indexing, one movie at a time."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import modal

app = modal.App("curtain-50k-collection-indexing")
volume = modal.Volume.from_name("curtain-training-data")
CHECKPOINT_HASH = "6058d992caa3f5c84696b63bc455e680a67e72f60d13280aad408d4288f29807"
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("numpy>=2.2,<3", "pillow>=11,<13", "torch>=2.8,<3", "torchvision>=0.23,<1")
         .add_local_dir("curtain_ml", remote_path="/root/curtain_ml", copy=True))


@app.function(image=image, gpu="L4", cpu=8, memory=16384, timeout=3600,
              volumes={"/data": volume}, max_containers=1)
def encode_movie(movie: str, count: int, source_hash: str,
                 collection: str, source_root: str) -> dict:
    import numpy as np
    from curtain_ml.indexing import build_movie_index

    volume.reload()
    checkpoint = Path("/data/runs/l4-50k/best.pt")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_HASH:
        raise RuntimeError("Unexpected checkpoint; refusing to mix models")
    output = Path("/data/indexes") / collection / movie
    with tempfile.TemporaryDirectory(prefix="curtain-index-") as tmp:
        source = Path(tmp) / movie
        source.mkdir()
        print(f"Extracting {movie}: {count} frames", flush=True)
        with tarfile.open(f"/data/datasets/{collection}/{movie}.tar") as archive:
            archive.extractall(source, filter="data")
        manifest = build_movie_index(checkpoint, source, output, batch_size=512,
                                     workers=8, device_name="cuda")
        matrix = np.load(output / "embeddings.npy")
        if matrix.shape != (count, 128) or not np.isfinite(matrix).all():
            raise RuntimeError("Index failed shape/finite validation")
        manifest.update(checkpoint_sha256=CHECKPOINT_HASH, source_sha256=source_hash,
                        source_root=source_root, sample_fps=2)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(source: str = "/home/koushik/curtain/screenshots",
         collection: str = "l4-50k") -> None:
    import numpy as np

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", collection):
        raise ValueError("Collection must contain only letters, numbers, underscores or hyphens")
    root = Path(source).expanduser().resolve(strict=True)
    destination = Path("/home/koushik/curtain/trained_indexes") / collection
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    summary = []
    for movie in sorted(root.iterdir()):
        if not movie.is_dir():
            continue
        paths = sorted(p for p in movie.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        if not paths:
            print(f"Skipping empty directory: {movie.name}", flush=True)
            continue
        local = destination / movie.name
        manifest_path = local / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            records = json.loads((local / "records.json").read_text())
            matrix = np.load(local / "embeddings.npy", mmap_mode="r")
            if (saved.get("checkpoint_sha256") == CHECKPOINT_HASH
                    and saved.get("source_root") == str(root)
                    and [r["filename"] for r in records] == [p.name for p in paths]
                    and matrix.shape == (len(paths), 128) and np.isfinite(matrix).all()):
                summary.append(saved)
                print(f"Verified existing {movie.name}: {len(paths)}", flush=True)
                continue
            raise RuntimeError(f"Existing index mismatch: {local}")
        with tempfile.TemporaryDirectory(prefix="curtain-upload-") as tmp:
            archive_path = Path(tmp) / f"{movie.name}.tar"
            digest = hashlib.sha256()
            with tarfile.open(archive_path, "w") as archive:
                for path in paths:
                    digest.update(path.name.encode())
                    digest.update(hashlib.sha256(path.read_bytes()).digest())
                    archive.add(path, arcname=path.name, recursive=False)
            print(f"Uploading {movie.name}: {len(paths)} images, {archive_path.stat().st_size / 1e6:.1f} MB", flush=True)
            with volume.batch_upload(force=True) as upload:
                upload.put_file(archive_path, f"/datasets/{collection}/{movie.name}.tar")
        manifest = encode_movie.remote(movie.name, len(paths), digest.hexdigest(), collection, str(root))
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "curtain-training-data",
                        f"/indexes/{collection}/{movie.name}", str(destination)], check=True)
        records = json.loads((local / "records.json").read_text())
        matrix = np.load(local / "embeddings.npy", mmap_mode="r")
        if ([r["filename"] for r in records] != [p.name for p in paths]
                or matrix.shape != (len(paths), 128) or not np.isfinite(matrix).all()):
            raise RuntimeError(f"Downloaded index verification failed: {movie.name}")
        summary.append(manifest)
        report = {"checkpoint_sha256": CHECKPOINT_HASH, "source_root": str(root), "movies": summary,
                  "total_frames": sum(m["frames"] for m in summary)}
        (destination / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"COMPLETE {movie.name}; {len(summary)} movies, {report['total_frames']} frames locally verified", flush=True)
    print(f"ALL COMPLETE: {len(summary)} movies, {sum(m['frames'] for m in summary)} frames", flush=True)
