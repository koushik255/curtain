"""Re-index every catalog movie with a new model on another machine's GPU.

In batches of --batch movies: copy their frames over ssh (the originals where
they are still whole, else the 640 px WebP copies in .local/frames-small/),
run rcmindex there, copy the arrays back into --out/batch-NNN/, and delete
the remote copies (never the local frames). The next batch copies while the
current one indexes. Resumable: movies whose report is already in --out are
skipped. Then build an index from --out with build_faiss.py --runs.

    .venv/bin/python -m rcmindex.reindex_remote --host koush@100.120.236.103 \\
        --model .local/rcmindex-model/colour-ft-v1 --out .local/rcmindex-runs-colour-ft-v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from rcmindex.search import FAISS_DIR
from rcmindex.validation_suite import LOCAL, SOURCES

SMALL = LOCAL / "frames-small"
REMOTE = "reindex"  # under the remote home: frames/<collection>/<movie>, out/batch-NNN
started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def source(movie: str) -> Path:
    collection, name = movie.split("/")
    small = SMALL / collection / name
    return small if small.is_dir() else SOURCES[collection] / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="ssh destination with rcmindex built for its GPU")
    parser.add_argument("--model", type=Path, required=True, help="exported model dir (same path on the host)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--rcmindex", default="curtain/rcmindex/build-cuda/rcmindex", help="path on the host")
    args = parser.parse_args()
    catalog = json.loads((FAISS_DIR / "catalog.json").read_text())
    done = {f"{p.parent.name}/{p.stem}" for p in args.out.glob("*/*/*.json") if not p.name.endswith(".records.json")}
    movies = [m["movie"] for m in catalog["movies"] if m["movie"] not in done]
    batches = [movies[i:i + args.batch] for i in range(0, len(movies), args.batch)]
    first = len(list(args.out.glob("batch-*")))
    say(f"{len(movies)} movies to index ({len(done)} done) in {len(batches)} batches")

    def ssh(command: str, stdin: str | None = None) -> None:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, command], input=stdin, text=True, check=True)

    class Copy(threading.Thread):
        """Copies one batch's frames in the background; join() re-raises its error."""

        def __init__(self, batch: list[str]):
            super().__init__()
            self.batch, self.error = batch, None
            self.start()

        def run(self) -> None:
            try:
                for movie in self.batch:
                    subprocess.run(["rsync", "-a", "--mkpath", f"{source(movie)}/",
                                    f"{args.host}:{REMOTE}/frames/{movie}/"], check=True)
            except Exception as error:  # noqa: BLE001 - reported on join
                self.error = error

        def join(self, timeout: float | None = None) -> None:
            super().join(timeout)
            if self.error:
                raise self.error

    copier = Copy(batches[0]) if batches else None
    for k, batch in enumerate(batches):
        copier.join()
        if k + 1 < len(batches):
            copier = Copy(batches[k + 1])
        name = f"batch-{first + k:03d}"
        say(f"{name}: indexing {len(batch)} movies ({batch[0]} ...)")
        dirs = "".join(f"{REMOTE}/frames/{movie}\n" for movie in batch)  # rcmindex reads movie dirs on stdin
        ssh(f"~/{args.rcmindex} --model ~/curtain/{args.model} --movies-from - --global "
            f"--output ~/{REMOTE}/out/{name} > /dev/null", stdin=dirs)
        (args.out / name).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{args.host}:{REMOTE}/out/{name}/", f"{args.out / name}/"], check=True)
        ssh(f"rm -rf ~/{REMOTE}/out/{name} " + " ".join(f"'{REMOTE}/frames/{movie}'" for movie in batch))
        frames = sum(json.loads(p.read_text())["frames"] for p in (args.out / name).glob("*/*.json")
                     if not p.name.endswith(".records.json"))
        say(f"{name}: done, {frames:,} frames")
    say("all movies indexed")


if __name__ == "__main__":
    main()
