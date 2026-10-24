"""Train on every film whose frames are on the curtain-reindex Modal volume.

rcmindex/reindex_modal.py left one tar of 640 px WebP frames per film on the
volume (frames/<collection>/<movie>.tar): 413 films, none of the validation
suite's six, which stay out of training. A container samples --frames
anchors per film, each with a frame 1..--neighbours later (hard negatives,
see GpuTrainingConfig.neighbours), unpacks just those to local disk and runs
curtain_ml.gpu_training on an A100 with the hard augmentations. The run goes
to the volume (runs/<name>/); --download copies it into curtain_models/<name>.

    .venv/bin/python -m modal run --detach curtain_ml/train_modal.py --name all-films-v1
    .venv/bin/python -m modal run curtain_ml/train_modal.py --name all-films-v1 --download
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "curtain-reindex"
INIT = "colour-ft-v1"  # curtain_models/<INIT>/best.pt, fine-tuned from

app = modal.App("curtain-train")
volume = modal.Volume.from_name(VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy>=2.2,<3", "pillow>=11,<13")
    .add_local_python_source("curtain_ml")
)


@app.function(image=image, gpu="A100-80GB", cpu=16, memory=65536, timeout=6 * 60 * 60,
              volumes={"/vol": volume})
def train_all(name: str, frames: int, reach: int, epochs: int, limit: int | None,
              batch_size: int, accumulate: int, learning_rate: float) -> dict:
    import io
    import random
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    from curtain_ml.gpu_training import GpuTrainingConfig, train

    started = time.monotonic()
    data = Path("/tmp/data")
    tars = sorted(Path("/vol/frames").glob("*/*.tar"))

    def unpack(tar_path: Path) -> int:
        # frames anchors, each with a frame 1..reach later, by position in the
        # film. The tar is read in one go: listing its ~17k members straight
        # from the volume is one small read each (22 min for 413 films).
        rng = random.Random(tar_path.stem)
        with tarfile.open(fileobj=io.BytesIO(tar_path.read_bytes())) as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
            anchors = rng.sample(range(len(members) - reach), min(frames, len(members) - reach))
            keep = {i for a in anchors for i in (a, a + rng.randint(1, reach))}
            target = data / tar_path.stem
            target.mkdir(parents=True, exist_ok=True)
            for i in sorted(keep):
                (target / Path(members[i].name).name).write_bytes(tar.extractfile(members[i]).read())
        return len(keep)

    with ThreadPoolExecutor(32) as pool:
        count = sum(pool.map(unpack, tars))
    print(f"unpacked {count:,} frames of {len(tars)} films in {time.monotonic() - started:.0f}s", flush=True)

    output = Path("/vol/runs") / name
    config = GpuTrainingConfig(
        data_dir=str(data), output_dir=str(output), epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, workers=14, limit=limit, forward_chunk_size=1024,
        evaluation_interval=2, init=f"/vol/models/{INIT}/best.pt", accumulate=accumulate,
        neighbours=reach, hard=True,
    )
    metrics = train(config)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(name: str, frames: int = 500, neighbours: int = 3, epochs: int = 6, limit: int = 0,
         batch_size: int = 256, accumulate: int = 2, learning_rate: float = 1e-4,
         download: bool = False) -> None:
    if download:
        target = ROOT / "curtain_models" / name
        target.mkdir(parents=True, exist_ok=True)
        for entry in volume.listdir(f"runs/{name}"):
            with open(target / Path(entry.path).name, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
        print(f"downloaded runs/{name} to {target}")
        return
    try:
        volume.listdir(f"models/{INIT}")
    except (modal.exception.NotFoundError, FileNotFoundError):
        with volume.batch_upload() as batch:
            batch.put_file(str(ROOT / "curtain_models" / INIT / "best.pt"), f"/models/{INIT}/best.pt")
    print(train_all.remote(name, frames, neighbours, epochs, limit or None, batch_size, accumulate,
                           learning_rate))
