from __future__ import annotations

import modal


app = modal.App("curtain-contrastive-training")
volume = modal.Volume.from_name("curtain-training-data", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.2,<3",
        "pillow>=11,<13",
        "torch>=2.8,<3",
        "torchvision>=0.23,<1",
    )
    .add_local_dir("training", remote_path="/root/training", copy=True)
)


@app.function(
    image=training_image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=2 * 60 * 60,
    retries=1,
    volumes={"/data": volume},
)
def train_on_l4(
    epochs: int = 20,
    batch_size: int = 128,
    image_size: int = 192,
    workers: int = 8,
    run_name: str = "l4-baseline",
    data_name: str = "selected_screenshots",
    archive_name: str | None = None,
    limit: int | None = None,
    lighting_preset: str = "legacy",
) -> dict[str, float]:
    import tarfile
    from pathlib import Path

    from training.train import TrainingConfig, train

    data_dir = f"/data/{data_name}"
    if archive_name is not None:
        extracted = Path("/tmp/curtain-training-data")
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f"/data/datasets/{archive_name}") as archive:
            archive.extractall(extracted, filter="data")
        data_dir = str(extracted)

    metrics = train(
        TrainingConfig(
            data_dir=data_dir,
            output_dir=f"/data/runs/{run_name}",
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            workers=workers,
            limit=limit,
            device="cuda",
            lighting_preset=lighting_preset,
        )
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 128,
    image_size: int = 192,
    workers: int = 8,
    run_name: str = "l4-smoke",
    data_name: str = "selected_screenshots",
    archive_name: str | None = None,
    limit: int | None = None,
    lighting_preset: str = "legacy",
) -> None:
    metrics = train_on_l4.remote(
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        workers=workers,
        run_name=run_name,
        data_name=data_name,
        archive_name=archive_name,
        limit=limit,
        lighting_preset=lighting_preset,
    )
    print(metrics)
