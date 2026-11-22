"""Train on every film whose frames are on the curtain-reindex Modal volume.

rcmindex/reindex_modal.py left one tar of 640 px WebP frames per film on the
volume (indexed_movies_640p/<collection>/<movie>.tar): 413 films, none of the
validation suite's six, which stay out of training. Each run samples --frames
anchors per film, each with a frame 1..--neighbours later (hard negatives, see
GpuTrainingConfig.neighbours), and trains curtain_ml.gpu_training with the
hard augmentations. --init picks the checkpoint it fine-tunes from
(curtain_models/<init>/best.pt, uploaded once); --tight trains on smaller,
thinner tile crops.

Three steps, so a GPU only ever trains:
  1. prepare_films (CPU, many containers at once) writes the sampled frames of
     each film to datasets/f<frames>-r<neighbours>/<movie>.tar, once per
     setting; later runs with the same setting reuse it.
  2. train_all (GPU) unpacks that dataset to local disk and trains. Each epoch
     ends with training_runs/<name>/last.pt and a metrics.jsonl line, committed
     to the volume; a preempted or crashed container is retried and resumes
     after the last finished epoch. Re-running a name with the same settings
     resumes it too (with other settings it refuses).
  3. orchestrate (CPU) runs 1 then 2 on the Modal side, so --detach survives
     closing the laptop at any point.

    .venv/bin/python -m modal run --detach curtain_ml/train_modal.py::main --name all-films-v2
    .venv/bin/python -m modal run --detach curtain_ml/train_modal.py::main --name tight-v2 \\
        --init all-films-v1 --tight --frames 1500 --epochs 4 --accumulate 4
    .venv/bin/python -m modal run --detach curtain_ml/train_modal.py::sweep --file runs.json --gpu L40S
    .venv/bin/python -m modal run curtain_ml/train_modal.py::main --name profile-l40s --gpu L40S \\
        --profile-steps 20 --max-dollars 2   # where a step's time goes: training_runs/<name>/profile.txt
    .venv/bin/python -m modal run curtain_ml/train_modal.py::report
    .venv/bin/python -m modal run curtain_ml/train_modal.py::main --name all-films-v2 --download

A sweep file is a JSON list of runs, each a name plus any of DEFAULTS to
change, and optionally its own "gpu":
    [{"name": "lr-3e-4", "learning_rate": 3e-4}, {"name": "tight", "tight": true}]
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "curtain-reindex"
SOURCES = "indexed_movies_640p"  # <collection>/<movie>.tar, every frame
UNINDEXED = "unindexed_movies_640p"  # <movie>.tar: the films indexed later (index_unindexed_modal.py)
VALIDATION = {"Do_The_Right_Thing_1989", "Gladiator_2000", "The_Apartment_1960",
              "The_Handmaiden_2016", "The_King_Of_Comedy_1982", "The_Matrix_1999"}
TRAIN_GPU = "A100-80GB"
# Modal rates (2026-11): A100-80GB $2.50/h, $0.0473/core/h, $0.008/GiB/h; train_all is 24 cores, 64 GiB.
TRAIN_DOLLARS_PER_HOUR = 2.50 + 24 * 0.0473 + 64 * 0.008
DEFAULTS = {
    "frames": 500, "neighbours": 3, "epochs": 6, "limit": 0, "batch_size": 256,
    "accumulate": 2, "learning_rate": 1e-4, "init": "colour-ft-v1", "tight": False,
    # all_films: also the films in UNINDEXED (840 instead of 468); web_colour:
    # GpuCombinedAugmenter's web regrades; web_pairs: a datasets/<name> from
    # curtain_ml/web_pairs.py (frames/ and queries/ tars: real web screenshots
    # warped onto their frames, as query copies).
    "all_films": False, "web_colour": False, "web_pairs": "",
    # A run's GPU time is capped at this many dollars (its timeout, no retries:
    # a preempted run stops and is resumed by launching it again).
    "max_dollars": 15.0,
    # Speed only (a run resumes with these changed): bf16 autocast, torch.compile,
    # and gradient caching only if the whole step doesn't fit (see GpuTrainingConfig).
    "precision": "bf16", "compile": True, "gradient_caching": "auto",
    # > 0: profile this many steps into training_runs/<name>/profile.txt and stop.
    "profile_steps": 0,
}

app = modal.App("curtain-train")
volume = modal.Volume.from_name(VOLUME)
cpu_image = modal.Image.debian_slim(python_version="3.12")
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy>=2.2,<3", "pillow>=11,<13")
    .add_local_python_source("curtain_ml")
)


def dataset_key(frames: int, neighbours: int, all_films: bool = False) -> str:
    return f"f{frames}-r{neighbours}" + ("-all" if all_films else "")


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=60 * 60, volumes={"/vol": volume},
              retries=2)
def prepare_films(sources: list[str], frames: int, reach: int, key: str) -> int:
    """Write the training sample of each film (a <movie>.tar path under /vol)
    to datasets/<key>/<movie>.tar: frames anchors, each with a frame
    1..reach later, by position in the film. Seeded by the film, so the same
    frames every time. Films already written are skipped; returns frames written."""
    import io
    import random
    import shutil
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    target_dir = Path("/vol/datasets") / key
    target_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp/prepare")
    scratch.mkdir(exist_ok=True)

    def prepare(source: str) -> int:
        source_path = Path("/vol") / source
        target = target_dir / f"{source_path.stem}.tar"
        if target.exists():
            return 0
        # Read the tar in one go: listing its ~17k members straight from the
        # volume is one small read each.
        rng = random.Random(source_path.stem)
        local = scratch / target.name
        with tarfile.open(fileobj=io.BytesIO(source_path.read_bytes())) as tar, \
                tarfile.open(local, "w") as out:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
            anchors = rng.sample(range(len(members) - reach), min(frames, len(members) - reach))
            keep = sorted({i for a in anchors for i in (a, a + rng.randint(1, reach))})
            for i in keep:
                member = members[i]
                data = tar.extractfile(member).read()
                info = tarfile.TarInfo(Path(member.name).name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(local, partial)
        partial.replace(target)
        local.unlink()
        return len(keep)

    with ThreadPoolExecutor(6) as pool:
        written = sum(pool.map(prepare, sources))
    volume.commit()
    return written


def ensure_dataset(frames: int, reach: int, all_films: bool = False) -> str:
    """Prepare whichever films of datasets/<key> are missing; returns the key."""
    import time

    key = dataset_key(frames, reach, all_films)
    volume.reload()
    paths = sorted(Path(f"/vol/{SOURCES}").glob("*/*.tar"))
    if all_films:  # a film in both (Redline_2009) is taken from SOURCES
        indexed = {p.stem for p in paths}
        paths += sorted(p for p in Path(f"/vol/{UNINDEXED}").glob("*.tar") if p.stem not in indexed)
    sources = [str(p.relative_to("/vol")) for p in paths if p.stem not in VALIDATION]
    done = {p.name for p in Path(f"/vol/datasets/{key}").glob("*.tar")}
    missing = [s for s in sources if f"{Path(s).stem}.tar" not in done]
    if missing:
        started = time.monotonic()
        chunks = [missing[i::32] for i in range(min(32, len(missing)))]
        written = sum(prepare_films.starmap((chunk, frames, reach, key) for chunk in chunks))
        volume.reload()
        print(f"dataset {key}: wrote {written:,} frames of {len(missing)} films "
              f"in {time.monotonic() - started:.0f}s", flush=True)
    else:
        print(f"dataset {key}: all {len(sources)} films ready", flush=True)
    return key


@app.function(image=gpu_image, gpu=TRAIN_GPU, cpu=24, memory=65536, timeout=6 * 60 * 60,
              volumes={"/vol": volume}, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train_all(name: str, options: dict) -> dict:
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    from curtain_ml.gpu_training import GpuTrainingConfig, train

    started = time.monotonic()
    volume.reload()
    key = dataset_key(options["frames"], options["neighbours"], options["all_films"])
    tars = sorted(Path(f"/vol/datasets/{key}").glob("*.tar"))
    if not tars:
        raise RuntimeError(f"datasets/{key} is empty; run through orchestrate, which prepares it")
    data = Path("/tmp/data")

    def unpack(tar_path: Path) -> int:
        target = data / tar_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(target, filter="data")
            return len(tar.getmembers())

    with ThreadPoolExecutor(32) as pool:
        count = sum(pool.map(unpack, tars))
    print(f"unpacked {count:,} frames of {len(tars)} films ({key}) in {time.monotonic() - started:.0f}s",
          flush=True)
    degraded = None
    if options["web_pairs"]:  # each web screenshot's frame (and neighbours) joins its film's folder
        pairs = Path("/vol/datasets") / options["web_pairs"]
        degraded = Path("/tmp/webq")
        for folder, target in (("frames", data), ("queries", degraded)):
            for tar_path in sorted((pairs / folder).glob("*.tar")):
                (target / tar_path.stem).mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_path) as tar:
                    tar.extractall(target / tar_path.stem, filter="data")
        print(f"web pairs {options['web_pairs']}: {sum(1 for _ in degraded.glob('*/*.jpg'))} screenshots",
              flush=True)

    config = GpuTrainingConfig(
        data_dir=str(data), output_dir=str(Path("/vol/training_runs") / name),
        epochs=options["epochs"], batch_size=options["batch_size"],
        learning_rate=options["learning_rate"], workers=22, limit=options["limit"] or None,
        forward_chunk_size=2048, evaluation_interval=2,
        init=f"/vol/trained_models_pt/{options['init']}/best.pt", accumulate=options["accumulate"],
        neighbours=options["neighbours"], hard=True, tight=options["tight"],
        web_colour=options["web_colour"], degraded=str(degraded) if degraded else None,
        degrade_chance=1.0, degraded_repeat=4 if degraded else 1,
        precision=options["precision"], compile=options["compile"],
        gradient_caching=options["gradient_caching"], profile_steps=options["profile_steps"],
    )
    metrics = train(config, on_checkpoint=volume.commit)
    volume.commit()
    return metrics


# Non-preemptible: a preempted orchestrate is restarted and spawns its runs
# again while the first ones keep going (two GPUs on one run). At 0.25 cores
# the higher rate costs next to nothing.
@app.function(image=cpu_image, cpu=0.25, timeout=24 * 60 * 60, volumes={"/vol": volume},
              nonpreemptible=True)
def orchestrate(runs: list[dict]) -> dict[str, dict | str]:
    """Prepare each run's dataset, then train every run at once, each on its
    "gpu" (default TRAIN_GPU). Returns each run's final metrics or its error."""
    for frames, neighbours, all_films in sorted({(r["frames"], r["neighbours"], r["all_films"]) for r in runs}):
        ensure_dataset(frames, neighbours, all_films)
    calls = {}
    for run in runs:
        options = {k: run[k] for k in DEFAULTS}
        gpu = run.get("gpu") or TRAIN_GPU
        timeout = int(run["max_dollars"] / TRAIN_DOLLARS_PER_HOUR * 3600)
        calls[run["name"]] = train_all.with_options(gpu=gpu, timeout=timeout, retries=0).spawn(run["name"], options)
        print(f"started {run['name']} on {gpu}, capped at ${run['max_dollars']:.2f} ({timeout / 3600:.1f} h): "
              f"{calls[run['name']].object_id}", flush=True)
    results: dict[str, dict | str] = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as error:  # one failed run doesn't lose the others' results
            results[name] = f"failed: {error!r}"
        print(f"{name}: {json.dumps(results[name])}", flush=True)
    return results


def complete_runs(runs: list[dict]) -> list[dict]:
    """Fill each run's unset options from DEFAULTS; refuse unknown keys and names twice."""
    filled = []
    for run in runs:
        unknown = set(run) - set(DEFAULTS) - {"name", "gpu"}
        if "name" not in run or unknown:
            raise SystemExit(f"each run needs a name and only {sorted(DEFAULTS)} or gpu; got {run}")
        filled.append({**DEFAULTS, **run})
    names = [r["name"] for r in filled]
    if len(set(names)) != len(names):
        raise SystemExit(f"run names repeat: {names}")
    return filled


def upload_inits(runs: list[dict]) -> None:
    """Upload curtain_models/<init>/best.pt for each init the volume lacks."""
    for init in sorted({r["init"] for r in runs}):
        try:
            volume.listdir(f"trained_models_pt/{init}")
        except (modal.exception.NotFoundError, FileNotFoundError):
            with volume.batch_upload() as batch:
                batch.put_file(str(ROOT / "curtain_models" / init / "best.pt"), f"/trained_models_pt/{init}/best.pt")
            print(f"uploaded {init}/best.pt")


def launch(runs: list[dict]) -> None:
    runs = complete_runs(runs)
    upload_inits(runs)
    print(json.dumps(orchestrate.remote(runs), indent=2))


@app.local_entrypoint()
def main(name: str, frames: int = DEFAULTS["frames"], neighbours: int = DEFAULTS["neighbours"],
         epochs: int = DEFAULTS["epochs"], limit: int = DEFAULTS["limit"],
         batch_size: int = DEFAULTS["batch_size"], accumulate: int = DEFAULTS["accumulate"],
         learning_rate: float = DEFAULTS["learning_rate"], init: str = DEFAULTS["init"],
         tight: bool = DEFAULTS["tight"], gpu: str = TRAIN_GPU, max_dollars: float = DEFAULTS["max_dollars"],
         profile_steps: int = DEFAULTS["profile_steps"], download: bool = False) -> None:
    if download:
        target = ROOT / "curtain_models" / name
        target.mkdir(parents=True, exist_ok=True)
        for entry in volume.listdir(f"training_runs/{name}"):
            with open(target / Path(entry.path).name, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
        print(f"downloaded training_runs/{name} to {target}")
        return
    launch([{"name": name, "frames": frames, "neighbours": neighbours, "epochs": epochs, "limit": limit,
             "batch_size": batch_size, "accumulate": accumulate, "learning_rate": learning_rate,
             "init": init, "tight": tight, "gpu": gpu, "max_dollars": max_dollars,
             "profile_steps": profile_steps}])


@app.local_entrypoint()
def sweep(file: str, gpu: str = "L40S") -> None:
    """Train every run in a JSON list at once, each on its own GPU (--gpu unless the run sets one)."""
    runs = json.loads(Path(file).read_text())
    launch([{"gpu": gpu, **run} for run in runs])


@app.local_entrypoint()
def report(names: str = "") -> None:
    """Latest and best evaluated epoch of each training run (comma-separated --names, default all)."""
    wanted = [n for n in names.split(",") if n] or sorted(
        Path(e.path).name for e in volume.listdir("training_runs")
        if e.type == modal.volume.FileEntryType.DIRECTORY)
    print(f"{'run':<28}{'epoch':>7}{'latest':>9}{'best':>9}{'@':>4}{'global':>9}{'grid':>9}")
    for name in wanted:
        try:
            lines = b"".join(volume.read_file(f"training_runs/{name}/metrics.jsonl")).decode().splitlines()
        except (modal.exception.NotFoundError, FileNotFoundError):
            print(f"{name:<28}{'(no metrics.jsonl)':>20}")
            continue
        epochs = [json.loads(line) for line in lines if line]
        evaluated = [e for e in epochs if e.get("evaluated")]
        if not evaluated:
            print(f"{name:<28}{epochs[-1]['epoch'] if epochs else 0:>7}{'-':>9}")
            continue
        latest, best = evaluated[-1], max(evaluated, key=lambda e: e["selection_score"])
        print(f"{name:<28}{epochs[-1]['epoch']:>7}{latest['selection_score']:>9.4f}"
              f"{best['selection_score']:>9.4f}{best['epoch']:>4}"
              f"{best['global_exact_frame_top1']:>9.4f}{best['grid_exact_frame_top1']:>9.4f}")
