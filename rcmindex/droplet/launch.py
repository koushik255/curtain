"""One command: create an AMD GPU droplet, get movies onto it, index them on
the GPU as they arrive, fetch the results, and delete the droplet.

Movies come from Spaces (default: fast, inside DigitalOcean; only movies that
upload_frames.py has stored) or, with --source local, straight from this
machine over ssh (limited by home upload speed). Transfers start as soon as
the droplet accepts ssh, while it is still installing and building, and every
movie is indexed the moment it has arrived. The droplet is deleted when the
run ends, fails, or you press Ctrl-C.

    python3 rcmindex/droplet/launch.py                     # 5 random stored movies
    python3 rcmindex/droplet/launch.py --movies 0          # every stored movie not yet indexed
    python3 rcmindex/droplet/launch.py --repeat 60 --min-new 20
        # every 60 min, if 20+ new movies are in Spaces, index them; stops once all are done
    python3 rcmindex/droplet/launch.py --source local --movies 2 --seed 3
    python3 rcmindex/droplet/launch.py --keep              # leave the droplet running
    python3 rcmindex/droplet/launch.py --destroy           # delete a kept droplet

Results land in .local/rcmindex-runs/<time>/.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spaces  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".local/rcmindex-model/combined-rocm-50k-v1"
# MI325X first (what we've used); fall back to other gfx942/gfx950-class sizes.
SIZES = ["gpu-mi325x1-256gb", "gpu-mi300x1-192gb"]
REGIONS = ["atl1", "tor1", "nyc2", "nyc3", "sfo3", "ams3", "fra1", "lon1", "ric1"]
IMAGE = "digitaloceanai-rocmsoftware"
SOURCES = spaces.SOURCES
SSH_KEY = "59419092"  # curtain-training-codex == ~/.ssh/id_ed25519

SETUP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
echo "[setup] apt packages"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq migraphx-dev libturbojpeg0-dev cmake rsync >/tmp/apt.log 2>&1 || { tail -20 /tmp/apt.log; exit 1; }
echo "[setup] build rcmindex"
cd ~/curtain-rcmindex
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm >/dev/null
cmake --build build -j"$(nproc)" > build.log 2>&1 || { grep -E "error" build.log | head -20; exit 1; }
echo "[setup] GPU: $(rocminfo | grep -m1 -o 'gfx9[0-9a-f]*'), ROCm $(cat /opt/rocm/.info/version)"
"""

started = time.monotonic()


def say(message: str) -> None:
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def doctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["doctl", *args, "-o", "json"], capture_output=True, text=True, check=check)


def find(name: str) -> dict | None:
    droplets = json.loads(doctl("compute", "droplet", "list").stdout or "[]")
    return next((d for d in droplets if d["name"] == name), None)


def get(droplet_id: int) -> dict | None:
    # By ID: unlike the name listing, this is current right after create.
    result = doctl("compute", "droplet", "get", str(droplet_id), check=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def delete(droplet_id: int) -> None:
    """Delete by ID until the API answers 404, which is the real confirmation
    (the droplet list can keep showing a deleted droplet for a while)."""
    for _ in range(12):
        result = doctl("compute", "droplet", "delete", "-f", str(droplet_id), check=False)
        if result.returncode != 0 and "404" in result.stderr + result.stdout:
            say(f"droplet {droplet_id} deleted")
            return
        time.sleep(5)
    say(f"DELETE NOT CONFIRMED, run: doctl compute droplet delete -f {droplet_id}")


def public_ip(droplet: dict) -> str | None:
    return next((n["ip_address"] for n in droplet["networks"]["v4"] if n["type"] == "public"), None)


def create(name: str, sizes: list[str], regions: list[str], owned: list[int]) -> dict:
    for size in sizes:
        for region in regions:
            say(f"trying {size} in {region}")
            result = doctl("compute", "droplet", "create", name, "--size", size, "--region", region,
                           "--image", IMAGE, "--ssh-keys", SSH_KEY, check=False)
            if result.returncode == 0:
                droplet_id = json.loads(result.stdout)[0]["id"]
                owned.append(droplet_id)  # from here on, cleanup will delete it
                say(f"created {name} ({droplet_id}): {size} in {region}; booting (GPU droplets take a few minutes)")
                while True:
                    droplet = get(droplet_id)
                    if droplet and droplet["status"] == "active" and public_ip(droplet):
                        return droplet
                    time.sleep(10)
                    say(f"  status: {droplet['status'] if droplet else 'pending'}")
            reason = ((result.stderr + result.stdout).strip().splitlines() or ["?"])[-1]
            say(f"  unavailable: {reason[:150]}")
    sys.exit("no AMD GPU droplet available in any region; try again later")


def ssh(ip: str, command: str, script: str | None = None) -> None:
    # A multi-line script goes over stdin to `bash -s`, so newlines survive.
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", command],
                   input=script, text=True, check=True)


def pick_movies(count: int, seed: int | None) -> list[tuple[str, Path]]:
    movies = [(collection, d) for collection, root in SOURCES.items() if root.is_dir()
              for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("*.jpg"))]
    return random.Random(seed).sample(movies, min(count, len(movies)))


def folder_bytes(directory: Path) -> int:
    return sum(e.stat().st_size for e in __import__("os").scandir(directory) if e.is_file())


def upload(ip: str, movies: list[tuple[str, Path]], failed: list[str]) -> None:
    """Stream each movie folder over ssh (tar | ssh tar -x), one after another,
    then mark it complete so the droplet starts indexing it right away."""
    total = sum(folder_bytes(d) for _, d in movies)
    sent = 0
    for number, (collection, directory) in enumerate(movies, 1):
        size = folder_bytes(directory)
        remote = f"/scratch/frames/{collection}"
        receive = (f"mkdir -p {remote} && tar -x -C {remote} && "
                   f"touch {remote}/{directory.name}/.complete")
        pack = subprocess.Popen(["tar", "-c", "-C", str(directory.parent), directory.name],
                                stdout=subprocess.PIPE)
        push = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", receive],
                                stdin=subprocess.PIPE)
        start = last = time.monotonic()
        done = 0
        while chunk := pack.stdout.read(1 << 20):
            push.stdin.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 5:
                last = now
                rate = done / (now - start) / 1e6
                say(f"upload {number}/{len(movies)} {directory.name}: {min(done / size, 1):4.0%}  "
                    f"{rate:5.1f} MB/s  ({(sent + done) / 1e9:.2f}/{total / 1e9:.2f} GB total)")
        push.stdin.close()
        if pack.wait() or push.wait():
            say(f"upload of {directory.name} FAILED")
            failed.append(directory.name)
            continue
        sent += size
        elapsed = time.monotonic() - start
        say(f"uploaded {number}/{len(movies)} {directory.name}: {size / 1e9:.2f} GB in {elapsed:.0f}s "
            f"({size / elapsed / 1e6:.1f} MB/s) -> GPU")
    subprocess.run(["ssh", f"root@{ip}", "touch /scratch/frames/.uploads-done"], check=True)
    say("all uploads finished")


RUNS = ROOT / ".local/rcmindex-runs"


def indexed_movies() -> set[tuple[str, str]]:
    """(collection, movie) already indexed by an earlier run (report written last)."""
    return {(r.parent.name, r.stem) for r in RUNS.glob("*/*/*.json") if not r.name.endswith(".records.json")}


def repeat(args: argparse.Namespace, argv: list[str]) -> None:
    """Rerun this script every N minutes; each run indexes only new movies."""
    i = argv.index("--repeat")
    child = [sys.executable, __file__, *argv[:i], *argv[i + 2:]]  # same options minus --repeat N
    while True:
        s3 = spaces.client()
        stored = spaces.stored_movies(s3)
        new = stored - indexed_movies()
        upload_done = len(stored) >= len(spaces.local_movies())
        if new and (len(new) >= args.min_new or upload_done):
            say(f"{len(new)} new movies in Spaces -> starting a GPU run")
            subprocess.run(child + ["--movies", "0"])
        elif upload_done:
            say("upload finished and every movie is indexed; stopping")
            return
        else:
            say(f"{len(new)} new movies (waiting for {args.min_new}); checking again in {args.repeat} min")
        time.sleep(args.repeat * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="curtain-rcmindex")
    parser.add_argument("--size", action="append", help="size slug(s) to try, in order")
    parser.add_argument("--region", action="append", help="region(s) to try, in order")
    parser.add_argument("--source", choices=["spaces", "local"], default="spaces")
    parser.add_argument("--movies", type=int, default=5, help="random movies to index (0 = all)")
    parser.add_argument("--seed", type=int, help="pick the same random movies again")
    parser.add_argument("--limit", type=int, default=0, help="frames per movie (0 = all)")
    parser.add_argument("--keep", action="store_true", help="don't delete the droplet afterwards")
    parser.add_argument("--destroy", action="store_true")
    parser.add_argument("--all", action="store_true", help="also re-index movies indexed before")
    parser.add_argument("--repeat", type=int, metavar="MINUTES", help="keep running batches of new movies")
    parser.add_argument("--min-new", type=int, default=20, help="with --repeat: new movies needed per run")
    args = parser.parse_args()
    if args.repeat:
        spaces.ensure_boto3()
        repeat(args, sys.argv[1:])
        return
    if args.source == "spaces":
        spaces.ensure_boto3()
        s3 = spaces.client()
        stored = sorted(spaces.stored_movies(s3) - (set() if args.all else indexed_movies()))
        if not stored:
            say("no new movies in Spaces to index (all stored ones are done); nothing to do")
            return
        count = len(stored) if args.movies == 0 else min(args.movies, len(stored))
        picked = random.Random(args.seed).sample(stored, count)
        # Presigned links: the droplet can download these parts, nothing else.
        links = [" ".join([collection, movie, *spaces.part_links(s3, collection, movie)])
                 for collection, movie in picked]

    droplet = find(args.name)
    if args.destroy:
        if droplet:
            delete(droplet["id"])
        else:
            say(f"no droplet named {args.name}")
        return

    if not (MODEL / "trunk.onnx").exists():
        sys.exit(f"missing {MODEL}; export the model first (rcmindex/README.md)")
    if droplet:
        say(f"reusing {args.name} ({droplet['size_slug']} in {droplet['region']['slug']})")
    ip = None
    owned: list[int] = [droplet["id"]] if droplet else []
    try:
        if not droplet:
            droplet = create(args.name, args.size or SIZES, args.region or REGIONS, owned)
        ip = public_ip(droplet)
        say(f"ip {ip}; waiting for ssh")
        while subprocess.run(["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=5",
                              f"root@{ip}", "true"], capture_output=True).returncode:
            time.sleep(5)

        say("copying rcmindex and model")
        subprocess.run(["ssh", f"root@{ip}", "mkdir -p /scratch/frames /scratch/out"], check=True)
        subprocess.run(["rsync", "-a", "--exclude", "build", f"{ROOT}/rcmindex/", f"root@{ip}:curtain-rcmindex/"], check=True)
        # Includes the compiled MIGraphX program (*.mxr) from earlier runs, saving the compile.
        subprocess.run(["rsync", "-a", f"{MODEL}/", f"root@{ip}:rcmindex-model/"], check=True)

        failed: list[str] = []
        # Transfer while the droplet installs and builds; the GPU picks movies up as they land.
        if args.source == "spaces":
            say(f"{len(links)} movies from Spaces: " + ", ".join(line.split()[1] for line in links[:8])
                + (" ..." if len(links) > 8 else ""))
            uploader = threading.Thread(target=ssh, args=(ip, "python3 -u ~/curtain-rcmindex/droplet/fetch_spaces.py 8",
                                                          "\n".join(links) + "\n"), daemon=True)
        else:
            movies = pick_movies(args.movies or 10**9, args.seed)
            say("movies: " + ", ".join(d.name for _, d in movies))
            uploader = threading.Thread(target=upload, args=(ip, movies, failed), daemon=True)
        uploader.start()
        ssh(ip, "bash -s", SETUP)

        command = "python3 -u ~/curtain-rcmindex/droplet/run_index.py"
        if args.limit:
            command += f" -- --limit {args.limit}"
        if args.movies == 0:
            command = command.replace("run_index.py", "run_index.py --delete-frames")
        ssh(ip, command)  # GPU progress streams here, interleaved with transfer progress
        uploader.join()
        if failed:
            say(f"failed uploads: {', '.join(failed)}")

        results = RUNS / time.strftime("%Y%m%d-%H%M%S")
        results.mkdir(parents=True)
        say(f"copying results to {results}")
        subprocess.run(["rsync", "-a", f"root@{ip}:/scratch/out/", f"{results}/"], check=True)
        subprocess.run(["rsync", "-a", "--include", "*.mxr", "--exclude", "*", f"root@{ip}:rcmindex-model/", f"{MODEL}/"])
        say("done")
    finally:
        if args.keep:
            say(f"droplet kept: ssh root@{ip}; delete with --destroy")
        else:
            for droplet_id in owned:
                say(f"deleting droplet {droplet_id}")
                delete(droplet_id)


if __name__ == "__main__":
    main()
