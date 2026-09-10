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
    .add_local_dir("curtain_ml", remote_path="/root/curtain_ml", copy=True)
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
    run_name: str = "l4-crop",
    archive_name: str = "curtain-selected-screenshots-50k.tar",
) -> dict[str, float]:
    # Extract the training archive, train on one L4, and commit the run.
    import tarfile
    from pathlib import Path

    from curtain_ml.training import TrainingConfig, train

    data_dir = Path("/tmp/curtain-training-data")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(f"/data/datasets/{archive_name}") as archive:
        archive.extractall(data_dir, filter="data")

    metrics = train(
        TrainingConfig(
            data_dir=str(data_dir),
            output_dir=f"/data/runs/{run_name}",
            epochs=epochs,
            batch_size=128,
        )
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    epochs: int = 5,
    run_name: str = "l4-smoke",
    archive_name: str = "curtain-selected-screenshots-50k.tar",
) -> None:
    # Launch the Modal training function with the compact CLI configuration.
    metrics = train_on_l4.remote(
        epochs=epochs,
        run_name=run_name,
        archive_name=archive_name,
    )
    print(metrics)
