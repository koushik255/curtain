"""compare_models.py on a Modal GPU: old vs new encoder, without re-indexing.

Uploads the six validation films' frames (one tar each, to bench/ on the
curtain-reindex volume, once), the checkpoints, and the queries exactly as
the page uploads them; a GPU container embeds every frame of those films
the way the index does with each checkpoint and ranks each query's true
frame among its film's frames (compare_models' rule and report).

    .venv/bin/python -m modal run rcmindex/compare_models_modal.py \\
        --checkpoints curtain_models/colour-ft-v1/best.pt,curtain_models/all-films-v1/best.pt
"""

from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "curtain-reindex"

app = modal.App("curtain-compare")
volume = modal.Volume.from_name(VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy>=2.2,<3", "pillow>=11,<13",
                    "faiss-cpu>=1.9,<2", "opencv-python-headless>=4.10,<5")
    .add_local_python_source("curtain_ml", "rcmindex")
)


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, timeout=60 * 60,
              volumes={"/vol": volume})
def rank_all(movies: list[str], checkpoints: list[str], items: bytes) -> dict[str, list]:
    import numpy as np
    import torch
    from PIL import Image

    from curtain_ml.model import CurtainEncoder
    from curtain_ml.training import reference_transform
    from rcmindex import search
    from rcmindex.compare_models import frame_vectors

    device = torch.device("cuda")
    paths = {}
    for movie in movies:
        target = Path("/tmp/frames") / movie
        target.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(Path(f"/vol/bench/{movie}.tar").read_bytes())) as tar:
            tar.extractall(target, filter="data")
        paths[movie] = sorted(target.glob("*.jpg"))
    queries = [(kind, Image.open(io.BytesIO(png)).convert("RGB"), target)
               for kind, png, target in pickle.loads(items)]
    report = {}
    for checkpoint in checkpoints:
        model = CurtainEncoder.from_checkpoint(Path(f"/vol/{checkpoint}"), device).eval()
        vectors = {m: torch.from_numpy(frame_vectors(model, paths[m], device)).to(device) for m in movies}
        transform = reference_transform(model.image_size)
        rows = []
        for kind, image, target in queries:
            movie, name = target.split("/")[1:]
            with torch.inference_mode():
                q = model(torch.stack([transform(i) for i in [image, *search.pieces(image)]]).to(device)).half()
            best = (vectors[movie] @ q.T).amax(dim=1).float()  # [frames, q]
            score = torch.maximum(best[:, 0] - search.WHOLE_HANDICAP, best.mean(dim=1))
            index = [p.name for p in paths[movie]].index(name)
            rows.append((kind, int((score > score[index]).sum()) + 1, float(score[index])))
        report[checkpoint] = rows
        print(f"{checkpoint}: {sum(r[1] == 1 for r in rows)}/{len(rows)} first", flush=True)
    return report


@app.local_entrypoint()
def main(checkpoints: str, extra: int = 30, seed: int = 11) -> None:
    import numpy as np

    from rcmindex.benchmark_int8 import queries
    from rcmindex.validation_suite import SOURCES, VALIDATION, browser_upload

    def listing(path: str) -> set[str]:
        try:
            return {e.path for e in volume.listdir(path, recursive=True)}
        except (modal.exception.NotFoundError, FileNotFoundError):
            return set()
    there = listing("bench") | listing("models")
    with volume.batch_upload(force=True) as batch:
        for movie in VALIDATION:
            if f"bench/{movie}.tar" not in there:
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w") as tar:
                    for path in sorted((SOURCES["full-curtain"] / movie).glob("*.jpg")):
                        tar.add(path, arcname=path.name)
                buffer.seek(0)
                batch.put_file(buffer, f"/bench/{movie}.tar")
                print(f"queued {movie}", flush=True)
        names = []
        for checkpoint in checkpoints.split(","):
            path = Path(checkpoint)
            name = f"models/{path.parent.name}/{path.name}"
            if name not in there:
                batch.put_file(str(ROOT / path), "/" + name)
            names.append(name)
    print("uploaded", flush=True)

    items = []
    for kind, image, target in queries(extra, seed):
        png = io.BytesIO()
        browser_upload(image).save(png, format="PNG")
        items.append((kind, png.getvalue(), target))
    report = rank_all.remote(VALIDATION, names, pickle.dumps(items))

    labels = [n.split("/")[1] for n in names]
    kinds = list(dict.fromkeys(kind for kind, _, _ in items))
    print("\n| type | " + " | ".join(f"{n} #1" for n in labels) + " | "
          + " | ".join(f"{n} median rank" for n in labels) + " | "
          + " | ".join(f"{n} true score" for n in labels) + " |")
    print("|---|" + "---|" * (3 * len(names)))
    for kind in [*kinds, "ALL"]:
        cells = []
        for metric in ("first", "rank", "score"):
            for n in names:
                sub = [r for r in report[n] if kind in ("ALL", r[0])]
                cells.append(f"{sum(r[1] == 1 for r in sub)}/{len(sub)}" if metric == "first" else
                             f"{np.median([r[1] for r in sub]):.0f}" if metric == "rank" else
                             f"{np.mean([r[2] for r in sub]):.3f}")
        print(f"| {kind} | " + " | ".join(cells) + " |")
