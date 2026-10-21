"""Does a new rcmindex run (another GPU or backend) match the indexed arrays?

Compares every movie in a new run directory with the catalog's arrays for the
same movie: frame names must agree, and each tile and whole-frame vector is
compared by cosine. Same frames through the same model should agree to fp16
noise (cosine ~0.999+); from the 640 px WebP copies instead of the 1280 px
originals, expect lower but still high values.

    .venv/bin/python -m rcmindex.compare_runs .local/rcmindex-runs/<time>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from rcmindex.search import FAISS_DIR


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="rcmindex --output directory: <run>/<collection>/<movie>.npy")
    args = parser.parse_args()
    catalog = {m["movie"]: m for m in json.loads((FAISS_DIR / "catalog.json").read_text())["movies"]}
    for tiles in sorted(args.run.glob("*/*.npy")):
        if re.search(r"\.(global|g\d+)\.npy$", tiles.name):
            continue
        movie = f"{tiles.parent.name}/{tiles.stem}"
        old = catalog.get(movie)
        if old is None:
            print(f"{movie}: not in the catalog")
            continue
        new_names = json.loads(tiles.with_suffix(".records.json").read_text())
        old_names = json.loads(Path(old["tiles"]).with_suffix(".records.json").read_text())
        if new_names != old_names:
            print(f"{movie}: frame names differ ({len(new_names)} vs {len(old_names)} frames)")
            continue
        report = json.loads(tiles.with_suffix(".json").read_text())
        line = f"{movie}: {len(new_names)} frames at {report['frames_per_second']:.0f}/s on {report['gpu']}"
        for label, new, ref in (("tiles", tiles, old["tiles"]), ("whole", tiles.with_suffix(".global.npy"), old["global"])):
            if new.exists():
                c = cosines(np.load(new, mmap_mode="r"), np.load(ref, mmap_mode="r")).ravel()
                line += f"; {label} cosine mean {c.mean():.4f}, p1 {np.percentile(c, 1):.4f}, min {c.min():.4f}"
        print(line)


if __name__ == "__main__":
    main()
