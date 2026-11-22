"""A pool of codec-damaged training frames (curtain_ml.encodings).

`select` (where the frames are) picks runs of consecutive frames from each
training film and writes one file list per collection, for rsync
--files-from to the GPU machine. `encode` (there) encodes each run as one
clip with a random recipe and saves every frame, resized back to its
source's size, as <out>/<film>/<frame>.jpg (quality 95). gpu_training
--degraded <out> then crops the query views from these copies.

youtube_vp9 is held out of training, so the stacked suite can show whether
the model learns codec damage in general or only these encoders.

    .venv/bin/python -m curtain_ml.codec_pool select --out .local/codec-pool
    rsync -a --files-from=.local/codec-pool/l4-50k-lbfive.txt .local/data/frames/webp/l4-50k-lbfive gpu:codec/clean
    .venv/bin/python -m curtain_ml.codec_pool encode --clean ~/codec/clean --out ~/codec/encoded
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from curtain_ml import encodings

HELD_OUT = {"youtube_vp9"}
TRAINING_RECIPES = [name for name in encodings.RECIPES if name not in HELD_OUT]
# The six films of the validation suite; never in the pool.
VALIDATION = {"Do_The_Right_Thing_1989", "Gladiator_2000", "The_Apartment_1960",
              "The_Handmaiden_2016", "The_King_Of_Comedy_1982", "The_Matrix_1999"}


def number(path: Path) -> int:
    return int(re.findall(r"\d+", path.stem)[-1])


def select(frames_root: Path, split: Path, out: Path, runs: int, length: int, seed: int) -> None:
    rng = random.Random(seed)
    films = sorted(set(json.loads(split.read_text())["training_movies"]) - VALIDATION)
    out.mkdir(parents=True, exist_ok=True)
    lists, missing = {}, []
    for film in films:
        folder = next((c / film for c in sorted(frames_root.iterdir()) if (c / film).is_dir()), None)
        if folder is None:
            missing.append(film)
            continue
        frames = sorted(folder.iterdir(), key=number)
        # Start of each window of `length` consecutive frame numbers, then
        # `runs` of them spread over the film, never touching each other.
        numbers = [number(p) for p in frames]
        starts = [i for i in range(len(frames) - length) if numbers[i + length - 1] - numbers[i] == length - 1]
        rng.shuffle(starts)
        taken: list[int] = []
        for start in starts:
            if len(taken) == runs:
                break
            if all(abs(start - t) > length for t in taken):
                taken.append(start)
        lines = lists.setdefault(folder.parent.name, [])
        for start in sorted(taken):
            lines += [f"{film}/{p.name}" for p in frames[start:start + length]]
    for collection, lines in lists.items():
        (out / f"{collection}.txt").write_text("\n".join(lines) + "\n")
        print(f"{collection}: {len(lines)} frames")
    if missing:
        print(f"no frames for {len(missing)} films: {', '.join(missing[:10])}")


def clips(clean: Path) -> list[tuple[str, list[Path]]]:
    # Each film's maximal runs of consecutive frame numbers.
    out = []
    for folder in sorted(p for p in clean.iterdir() if p.is_dir()):
        run: list[Path] = []
        for path in sorted(folder.iterdir(), key=number):
            if run and number(path) != number(run[-1]) + 1:
                out.append((folder.name, run))
                run = []
            run.append(path)
        if run:
            out.append((folder.name, run))
    return out


def encode_clip(job: tuple[str, list[Path], Path, int]) -> tuple[str, int, str]:
    film, paths, out, seed = job
    rng = random.Random(f"{seed}/{film}/{paths[0].name}")
    recipe = rng.choice(TRAINING_RECIPES)
    folder = out / film
    if all((folder / f"{p.stem}.jpg").exists() for p in paths):
        return film, len(paths), "done"
    frames = [Image.open(p).convert("RGB") for p in paths]
    folder.mkdir(parents=True, exist_ok=True)
    for path, frame, damaged in zip(paths, frames, encodings.encode(recipe, frames, None, rng)):
        if damaged.size != frame.size:
            damaged = damaged.resize(frame.size, Image.BICUBIC)
        damaged.save(folder / f"{path.stem}.jpg", quality=95)
    return film, len(paths), recipe


def encode(clean: Path, out: Path, processes: int, seed: int) -> None:
    jobs = [(film, paths, out, seed) for film, paths in clips(clean)]
    print(f"{len(jobs)} clips, {sum(len(j[1]) for j in jobs)} frames", flush=True)
    started, frames, recipes = time.perf_counter(), 0, {}
    with Pool(processes) as pool:
        for n, (film, count, recipe) in enumerate(pool.imap_unordered(encode_clip, jobs), 1):
            frames += count
            recipes[recipe] = recipes.get(recipe, 0) + 1
            if n % 100 == 0 or n == len(jobs):
                rate = frames / (time.perf_counter() - started)
                print(f"  {n}/{len(jobs)} clips, {rate:.0f} frames/s", flush=True)
    print(json.dumps(recipes, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--frames-root", type=Path, default=Path(".local/data/frames/webp"))
    s.add_argument("--split", type=Path, default=Path("curtain_models/all-films-v1/split.json"))
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--runs", type=int, default=8, help="runs per film")
    s.add_argument("--length", type=int, default=48, help="frames per run (24 s at 2 fps)")
    s.add_argument("--seed", type=int, default=5)
    e = sub.add_parser("encode")
    e.add_argument("--clean", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--processes", type=int, default=10)
    e.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    if args.command == "select":
        select(args.frames_root, args.split, args.out, args.runs, args.length, args.seed)
    else:
        encode(args.clean, args.out, args.processes, args.seed)


if __name__ == "__main__":
    main()
