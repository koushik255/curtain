from __future__ import annotations

import modal


app = modal.App("curtain-movie-indexing")
volume = modal.Volume.from_name("curtain-training-data", create_if_missing=True)

indexing_image = (
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
    image=indexing_image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={"/data": volume},
)
def index_on_l4(movie_name: str, run_name: str = "l4-baseline") -> dict:
    from pathlib import Path

    from training.index_movie import build_movie_index

    manifest = build_movie_index(
        checkpoint_path=Path(f"/data/runs/{run_name}/best.pt"),
        movie_dir=Path(f"/data/full_movies/{movie_name}"),
        output_dir=Path(f"/data/indexes/{run_name}/{movie_name}"),
        device_name="cuda",
    )
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(movie_name: str = "The_Matrix_1999", run_name: str = "l4-baseline") -> None:
    print(index_on_l4.remote(movie_name=movie_name, run_name=run_name))
