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
    limit: int | None = None,
) -> dict[str, float]:
    from training.train import TrainingConfig, train

    metrics = train(
        TrainingConfig(
            data_dir="/data/selected_screenshots",
            output_dir=f"/data/runs/{run_name}",
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            workers=workers,
            limit=limit,
            device="cuda",
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
    limit: int | None = None,
) -> None:
    metrics = train_on_l4.remote(
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        workers=workers,
        run_name=run_name,
        limit=limit,
    )
    print(metrics)
