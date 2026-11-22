"""Index the films in curtain-reindex/unindexed_movies_640p/ on Modal L4s, within a budget.

The tars are already on the volume, so nothing is uploaded: batches of --batch
films go to reindex_modal.index_batch (same image, same rcmindex), at most
--parallel at once, and each finished batch is downloaded into --out/unindexed-NNN/.

Spending is estimated from container wall time (L4 + 8 CPU + 32 GiB, measured
from spawn, so queueing counts too: an overestimate). No batch starts unless
the spend so far plus every running batch's expected cost fits in --budget, and
running batches are cancelled if the estimate reaches it.

Resumable: films whose report is already in --out are skipped. Afterwards add
them to the index with build_faiss.py --runs.

    nohup .venv/bin/python -m rcmindex.index_unindexed_modal --budget 10 \\
        > .local/index-unindexed.log 2>&1 &
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import modal

from rcmindex.reindex_modal import app, index_batch, volume

ROOT = Path(__file__).resolve().parents[1]
# Modal rates (modal billing rates, 2026-11): L4 $0.80/h, $0.0473/core/h, $0.008/GiB/h.
DOLLARS_PER_HOUR = 0.80 + 8 * 0.0473 + 32 * 0.008
DUPLICATES = {"Redline_2009"}  # full-curtain/Redline_2009 is indexed
FULL_CURTAIN = {"Terminator_2_Judgment_Day_1991"}
started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} {time.monotonic() - started:6.0f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-films-v1-onnx", help="folder in trained_models_onnx/ on the volume")
    parser.add_argument("--out", type=Path, default=ROOT / ".local/rcmindex-runs-all-films-v1")
    parser.add_argument("--budget", type=float, required=True, help="dollars")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    films = sorted(e.path.split("/")[-1][:-4] for e in volume.listdir("unindexed_movies_640p")
                   if e.path.endswith(".tar"))
    movies = [("full-curtain/" if f in FULL_CURTAIN else "l4-50k-lbfive/") + f for f in films if f not in DUPLICATES]
    done = {f"{p.parent.name}/{p.stem}" for p in args.out.glob("*/*/*.json") if not p.name.endswith(".records.json")}
    todo = [m for m in movies if m not in done]
    batches = [(f"unindexed-{movies.index(b[0]):03d}", b)
               for b in (todo[i:i + args.batch] for i in range(0, len(todo), args.batch))]
    say(f"{len(movies)} unindexed films, {len(movies) - len(todo)} already done, "
        f"{len(batches)} batches of {args.batch}; budget ${args.budget:.2f} at ${DOLLARS_PER_HOUR:.3f}/container-hour")

    lock = threading.Lock()
    spent = 0.0  # finished batches
    running: dict[str, tuple[float, modal.FunctionCall]] = {}
    hours_per_batch = [0.5]  # first guess, then the average of finished batches
    finished_films = 0
    stop = threading.Event()

    def accrued() -> float:
        now = time.monotonic()
        return spent + sum((now - t) / 3600 * DOLLARS_PER_HOUR for t, _ in running.values())

    def watchdog() -> None:
        while not stop.wait(15):
            with lock:
                if accrued() >= args.budget:
                    say(f"budget reached (${accrued():.2f}): cancelling {len(running)} running batches")
                    for _, call in running.values():
                        call.cancel()
                    stop.set()

    def run(name: str, batch: list[str]) -> None:
        nonlocal spent, finished_films
        call = running[name][1]
        try:
            frames = call.get()
        except Exception as ex:
            with lock:
                spent += (time.monotonic() - running.pop(name)[0]) / 3600 * DOLLARS_PER_HOUR
            say(f"{name}: failed or cancelled ({type(ex).__name__}); spent ~${spent:.2f}")
            return
        with lock:
            hours = (time.monotonic() - running.pop(name)[0]) / 3600
            spent += hours * DOLLARS_PER_HOUR
            hours_per_batch.append(hours)
            finished_films += len(batch)
        local = args.out / name
        prefix = f"index_outputs/{args.model}/{name}/"
        for entry in volume.listdir(prefix, recursive=True):
            if entry.type != modal.volume.FileEntryType.FILE:
                continue
            path = local / entry.path.removeprefix(prefix)
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(path.suffix + ".part")
            with open(part, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
            part.rename(path)
        say(f"{name}: {len(batch)} films, {frames:,} frames in {hours * 60:.1f} min; "
            f"{finished_films} films done, spent ~${spent:.2f}")

    # Until the TensorRT engines are on the volume, run one batch alone so they
    # are built (and saved by index_batch) once rather than in every container.
    cached = any(e.path.endswith(".plan") for e in volume.listdir(f"trained_models_onnx/{args.model}"))
    threads: list[threading.Thread] = []
    with modal.enable_output(), app.run():
        threading.Thread(target=watchdog, daemon=True).start()
        for name, batch in batches:
            while True:
                with lock:
                    estimate = sum(hours_per_batch[1:] or hours_per_batch) / len(hours_per_batch[1:] or hours_per_batch)
                    room = args.budget - accrued() - len(running) * estimate * DOLLARS_PER_HOUR
                    limit = args.parallel if cached or finished_films else 1
                    if stop.is_set() or (len(running) < limit and room >= estimate * DOLLARS_PER_HOUR * 1.2):
                        break
                    if not running:  # nothing running and no room: the budget is used up
                        stop.set()
                        break
                time.sleep(5)
            if stop.is_set():
                say("not starting more batches: budget")
                break
            with lock:
                running[name] = (time.monotonic(), index_batch.spawn(name, batch, args.model))
            say(f"{name}: started ({batch[0].split('/')[1]} ...)")
            thread = threading.Thread(target=run, args=(name, batch))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        stop.set()
    say(f"finished: {finished_films} films indexed, estimated spend ${spent:.2f}")


if __name__ == "__main__":
    main()
