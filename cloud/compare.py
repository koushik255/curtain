from __future__ import annotations

import modal


app = modal.App("curtain-checkpoint-comparison")
volume = modal.Volume.from_name("curtain-training-data", create_if_missing=True)

comparison_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.2,<3",
        "pillow>=11,<13",
        "torch>=2.8,<3",
        "torchvision>=0.23,<1",
    )
    .add_local_dir("curtain_ml", remote_path="/root/curtain_ml", copy=True)
    .add_local_dir("benchmarks", remote_path="/root/benchmarks", copy=True)
)


@app.function(
    image=comparison_image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=60 * 60,
    volumes={"/data": volume},
)
def compare_on_l4() -> dict:
    import tarfile
    from pathlib import Path

    from benchmarks.compare_checkpoints import compare

    extracted = Path("/tmp/curtain-comparison-data")
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/data/datasets/curtain-selected-screenshots-50k.tar") as archive:
        archive.extractall(extracted, filter="data")
    result = compare(
        data_dir=extracted,
        checkpoints={
            "5k_baseline": Path("/data/runs/l4-baseline/best.pt"),
            "50k_model": Path("/data/runs/l4-50k/best.pt"),
        },
        output_path=Path("/data/comparisons/5k-vs-50k.json"),
        device_name="cuda",
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    result = compare_on_l4.remote()
    print(result["difference_50k_minus_5k"])
