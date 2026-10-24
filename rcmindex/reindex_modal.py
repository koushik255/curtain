"""Re-index every catalog movie with a new model on Modal L4 GPUs.

reindex_remote.py's job with Modal in place of a machine on the network. In
batches of --batch movies: pack each movie's frames into one tar (the 640 px
WebPs where they exist, else the originals; millions of small files upload
far slower than a few hundred tars) and upload them to the curtain-reindex
volume; then a GPU container unpacks them, runs rcmindex (CUDA + TensorRT,
built in the image; L4 is Ada like the RTX 4060) and writes the arrays to
the volume, while the next batch uploads. Finished batches are downloaded
into --out/modal-NNN/ (NNN: the first movie's position in the catalog).

Resumable: movies whose report is already in --out are skipped, and tars
already on the volume are not uploaded again. The local frames are never
touched. Afterwards delete the volume (modal volume delete curtain-reindex)
and build an index from --out with build_faiss.py --runs.

    nohup .venv/bin/python -m rcmindex.reindex_modal --model curtain_models/colour-ft-v1-onnx \\
        --out .local/rcmindex-runs-colour-ft-v1 > .local/reindex-modal.log 2>&1 &
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "curtain-reindex"  # frames/<collection>/<movie>.tar, model/, out/<batch>/
UPLOADS = 4  # movies uploading at once

app = modal.App("curtain-reindex")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:25.01-py3")
    .apt_install("libturbojpeg0-dev", "libwebp-dev")
    .run_commands("python3 -m pip install -U 'cmake>=3.28,<4'")  # the image's is 3.24
    .add_local_dir(ROOT / "rcmindex/src", "/rcmindex/src", copy=True)
    .add_local_file(ROOT / "rcmindex/CMakeLists.txt", "/rcmindex/CMakeLists.txt", copy=True)
    .run_commands("cmake -S /rcmindex -B /rcmindex/build -DRCM_BACKEND=cuda -DCMAKE_BUILD_TYPE=Release"
                  " && cmake --build /rcmindex/build -j")
)


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=2 * 60 * 60,
              volumes={"/vol": volume}, retries=1)
def index_batch(name: str, movies: list[str], model: str) -> int:
    """Index movies ("<collection>/<movie>") into /vol/out/<name>; returns frames indexed."""
    import shutil

    volume.reload()
    work = Path("/tmp/work")
    shutil.rmtree(work, ignore_errors=True)
    for movie in movies:
        target = work / "frames" / movie
        target.mkdir(parents=True)
        with tarfile.open(f"/vol/frames/{movie}.tar") as tar:
            tar.extractall(target, filter="data")
    # rcmindex caches its TensorRT engine beside the model, so a local copy.
    shutil.copytree(f"/vol/model/{model}", work / "model")
    dirs = "".join(f"{work / 'frames' / movie}\n" for movie in movies)
    out = work / "out"
    subprocess.run(["/rcmindex/build/rcmindex", "--model", str(work / "model"), "--movies-from", "-",
                    "--global", "--output", str(out)], input=dirs, text=True, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copytree(out, f"/vol/out/{name}", dirs_exist_ok=True)
    volume.commit()
    return sum(json.loads(p.read_text())["frames"] for p in out.glob("*/*.json")
               if not p.name.endswith(".records.json"))


started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def main() -> None:
    from rcmindex.reindex_remote import source
    from rcmindex.search import FAISS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="exported model dir (trunk.onnx, head.*)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--limit", type=int, help="only this many batches (a test run)")
    args = parser.parse_args()
    catalog = [m["movie"] for m in json.loads((FAISS_DIR / "catalog.json").read_text())["movies"]]
    done = {f"{p.parent.name}/{p.stem}" for p in args.out.glob("*/*/*.json") if not p.name.endswith(".records.json")}
    movies = [m for m in catalog if m not in done]
    batches = [movies[i:i + args.batch] for i in range(0, len(movies), args.batch)][:args.limit]
    say(f"{len(movies)} movies to index ({len(done)} done) in {len(batches)} batches")

    def listing(path: str) -> list:
        try:
            return volume.listdir(path, recursive=True)
        except (modal.exception.NotFoundError, FileNotFoundError):
            return []
    uploaded = {e.path.removeprefix("frames/") for e in listing("frames") if e.path.endswith(".tar")}
    model = args.model.name
    if not listing(f"model/{model}"):
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(args.model), f"/model/{model}")
        say(f"uploaded model {model}")

    def upload(movie: str) -> int:
        if f"{movie}.tar" in uploaded:
            return 0
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:  # WebP/JPEG: no point compressing
            for path in sorted(source(movie).iterdir()):
                if path.suffix in (".webp", ".jpg"):
                    tar.add(path, arcname=path.name)
        size = buffer.tell()
        buffer.seek(0)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(buffer, f"/frames/{movie}.tar")
        return size

    def download(name: str, call: modal.FunctionCall) -> None:
        frames = call.get()
        local = args.out / name
        for entry in volume.listdir(f"out/{name}", recursive=True):
            if entry.type != modal.volume.FileEntryType.FILE:
                continue
            path = local / entry.path.removeprefix(f"out/{name}/")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_suffix(path.suffix + ".part"), "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
            path.with_suffix(path.suffix + ".part").rename(path)
        say(f"{name}: downloaded, {frames:,} frames")

    downloads: list[threading.Thread] = []
    pool = ThreadPoolExecutor(UPLOADS)
    with modal.enable_output(), app.run(detach=True):
        for batch in batches:
            name = f"modal-{catalog.index(batch[0]):03d}"
            t = time.monotonic()
            sizes = list(pool.map(upload, batch))
            mb = sum(sizes) / 1e6
            say(f"{name}: uploaded {len(batch)} movies, {mb:,.0f} MB in {time.monotonic() - t:.0f}s"
                f" ({mb / max(time.monotonic() - t, 1):.1f} MB/s)")
            call = index_batch.spawn(name, batch, model)
            thread = threading.Thread(target=download, args=(name, call))
            thread.start()
            downloads.append(thread)
        for thread in downloads:
            thread.join()
    say("all movies indexed")


if __name__ == "__main__":
    main()
