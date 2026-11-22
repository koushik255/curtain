"""Scene model: name the film from a moment that isn't in our footage.

The frame model (curtain_ml/gpu_training.py) finds the exact frame: it is
trained so that even neighbouring frames of one shot score apart. Most web
images whose film is not in the site's 20 results are moments we don't have
at all (publicity stills, other angles, colour copies of black-and-white
films; benchmarks/web_search_oracle.py, checked by eye). Our footage does
have the same scene: the same set, costumes, people and grade, a few seconds
or a minute away. This model is trained for that: DINOv2 ViT-S/14 at 224 px,
where a frame's positives are the frames of its film within WINDOW frames
(60 s at 2 fps), other films are negatives, and the same film's other scenes
are neither. The query side gets the frame model's hard augmentations plus
the web regrades (GpuCombinedAugmenter(web_colour=True)).

Data: datasets/f500-r3-all on the curtain-reindex volume (train_modal's
sample of ~1,000 frames of each of the 840 films), also the scene index the
evaluation searches (one vector a frame; a film's score is its best frame).

    .venv/bin/python -m modal run --detach curtain_ml/scene_modal.py::prepare
    .venv/bin/python -m modal run --detach curtain_ml/scene_modal.py::train --name scene-v1
    .venv/bin/python -m modal run curtain_ml/scene_modal.py::evaluate --model pretrained
    .venv/bin/python -m modal run curtain_ml/scene_modal.py::evaluate --model scene-v1

Each GPU function's timeout caps its spend (DOLLARS_PER_HOUR, max_dollars).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import modal

from curtain_ml.train_modal import app, dataset_key, ensure_dataset, volume

ROOT = Path(__file__).resolve().parents[1]
KEY = dataset_key(500, 3, True)
BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SIZE = 224
WINDOW = 120  # frames (2 fps): the same scene
# Modal rates (2026-11): A100-80GB $2.50/h, L4 $0.80/h, $0.0473/core/h, $0.008/GiB/h.
TRAIN_RATE = 2.50 + 24 * 0.0473 + 64 * 0.008
EMBED_RATE = 0.80 + 8 * 0.0473 + 32 * 0.008

scene_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "timm==1.0.15", "huggingface_hub>=0.30,<1", "safetensors>=0.5", "numpy>=2.2,<3",
                    "pillow>=11,<13")
    .env({"HF_HOME": "/root/hf"})  # not the volume: its open files block volume.reload
    .add_local_python_source("curtain_ml")
)


def frame_number(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def backbone(pretrained: bool = True):
    import timm
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0, img_size=SIZE)


def load_model(model: str, device):
    """'pretrained' (DINOv2 as released) or a training run's name."""
    import torch
    net = backbone(pretrained=model == "pretrained")
    if model != "pretrained":
        state = torch.load(f"/vol/training_runs/{model}/scene.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
    return net.to(device).eval()


def unpack(target: Path) -> dict[str, list[Path]]:
    """datasets/KEY/<film>.tar -> target/<film>/; returns each film's frames by number."""
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    tars = sorted(Path(f"/vol/datasets/{KEY}").glob("*.tar"))

    def one(tar_path: Path) -> tuple[str, list[Path]]:
        folder = target / tar_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(folder, filter="data")
        return tar_path.stem, sorted(folder.iterdir(), key=lambda p: frame_number(p.name))

    with ThreadPoolExecutor(32) as pool:
        return dict(pool.map(one, tars))


@app.function(image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("curtain_ml"), cpu=0.25, timeout=3 * 60 * 60,
              volumes={"/vol": volume})
def prepare_all() -> str:
    return ensure_dataset(500, 3, True)


@app.function(image=scene_image, gpu="A100-80GB", cpu=24, memory=65536, timeout=60 * 60,
              volumes={"/vol": volume})
def train_scene(name: str, options: dict) -> dict:
    import bisect
    import math
    import random
    import time

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from curtain_ml.gpu_augmentations import GpuCombinedAugmenter
    from curtain_ml.gpu_training import decode

    started = time.monotonic()
    volume.reload()
    films = unpack(Path("/tmp/data"))
    names = sorted(films)
    numbers = {f: [frame_number(p.name) for p in films[f]] for f in names}
    print(f"unpacked {sum(map(len, films.values())):,} frames of {len(names)} films "
          f"in {time.monotonic() - started:.0f}s", flush=True)

    def pairs_for(epoch: int) -> list[tuple[int, int, int]]:
        """(film, frame, partner): every frame with a random other frame of
        its film within WINDOW, by frame number."""
        rng = random.Random(epoch)
        out = []
        for f, name in enumerate(names):
            nums = numbers[name]
            for i, n in enumerate(nums):
                lo, hi = bisect.bisect_left(nums, n - WINDOW), bisect.bisect_right(nums, n + WINDOW)
                choices = [j for j in range(lo, hi) if j != i]
                if choices:
                    out.append((f, i, rng.choice(choices)))
        rng.shuffle(out)
        return out

    class Pairs(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, k):
            f, i, j = self.items[k]
            paths = films[names[f]]
            return decode(paths[i]), decode(paths[j]), f, numbers[names[f]][i], numbers[names[f]][j]

    def collate(batch):
        anchors, partners, f, a, b = zip(*batch)
        return (list(anchors) + list(partners), torch.tensor(f + f),
                torch.tensor(a + b, dtype=torch.long))

    device = torch.device("cuda")
    model = backbone().to(device)
    # Only the top half trains (and is checkpointed): ~1.6x faster, and keeps
    # more of what DINOv2 already knows.
    frozen = options["frozen_blocks"]
    for module in (model.patch_embed, *model.blocks[:frozen]):
        module.requires_grad_(False)
    model.cls_token.requires_grad_(False); model.pos_embed.requires_grad_(False)
    model.set_grad_checkpointing(True)
    augmenter = GpuCombinedAugmenter(SIZE, 4, hard=True, web_colour=True)
    pairs_per_batch = options["batch_pairs"]
    steps_per_epoch = len(pairs_for(0)) // pairs_per_batch
    total = steps_per_epoch * options["epochs"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=options["learning_rate"], weight_decay=0.05)
    warmup = min(300, total // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    out_dir = Path(f"/vol/training_runs/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if (out_dir / "last.pt").exists():
        state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); start_epoch = state["epoch"] + 1
        print(f"resuming after epoch {state['epoch']}", flush=True)
    temperature = options["temperature"]
    history = []
    for epoch in range(start_epoch, options["epochs"] + 1):
        loader = DataLoader(Pairs(pairs_for(epoch)), batch_size=pairs_per_batch, shuffle=False, drop_last=True,
                            num_workers=22, collate_fn=collate, prefetch_factor=2, persistent_workers=False)
        model.train()
        t0, total_loss, steps = time.monotonic(), 0.0, 0
        for images, film, number in loader:
            film, number = film.to(device), number.to(device)
            full, query = augmenter(images, device)[:2]  # index-side full frame, query-side crop
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = F.normalize(model(query).float(), dim=1)
                k = F.normalize(model(full).float(), dim=1)
            logits = q @ k.T / temperature
            same_film = film[:, None] == film[None, :]
            near = (number[:, None] - number[None, :]).abs() <= WINDOW
            positive = same_film & near
            counted = positive | ~same_film  # the same film's other scenes are left out
            logits = logits.masked_fill(~counted, float("-inf"))
            loss = -(torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
                     - torch.logsumexp(logits, dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); steps += 1
            if steps % 400 == 0:  # a timeout loses at most 400 steps
                torch.save({"model": model.state_dict(), "backbone": BACKBONE, "size": SIZE, "options": options,
                            "epoch": epoch - 1, "step": steps}, out_dir / "scene.pt")
                volume.commit()
            if steps % 50 == 0:
                print(f"  epoch {epoch}: step {steps}/{steps_per_epoch}, loss {total_loss / steps:.4f}, "
                      f"{steps * pairs_per_batch * 2 / (time.monotonic() - t0):.0f} frames/s", flush=True)
        history.append({"epoch": epoch, "loss": total_loss / max(1, steps), "seconds": time.monotonic() - t0})
        print(json.dumps(history[-1]), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch}, out_dir / "last.pt")
        torch.save({"model": model.state_dict(), "backbone": BACKBONE, "size": SIZE, "options": options,
                    "epoch": epoch}, out_dir / "scene.pt")
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(history[-1]) + "\n")
        volume.commit()
    return {"history": history}


@app.function(image=scene_image, gpu="L4", cpu=8, memory=32768, timeout=60 * 60, volumes={"/vol": volume})
def embed_and_rank(model_name: str, queries: dict[str, bytes], top: int = 100) -> dict:
    """Embed every frame of datasets/KEY (saved to scene_index/<model>/ and
    reused) and each query (a whole image, squashed to SIZE like the training
    full view); returns each query's best films [(film, score)] best first."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    device = torch.device("cuda")
    model = load_model(model_name, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def load(source) -> torch.Tensor:
        with Image.open(source) as im:
            im = im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

    @torch.inference_mode()
    def embed(batch: list[torch.Tensor]) -> np.ndarray:
        x = (torch.stack(batch).to(device).float() / 255 - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            return F.normalize(model(x).float(), dim=1).cpu().numpy().astype(np.float16)

    index_dir = Path(f"/vol/scene_index/{model_name}")
    volume.reload()
    if (index_dir / "vectors.npy").exists():
        vectors = np.load(index_dir / "vectors.npy")
        frames = json.loads((index_dir / "frames.json").read_text())
    else:
        started = time.monotonic()
        films = unpack(Path("/tmp/data"))
        frames = [(film, p.name) for film in sorted(films) for p in films[film]]
        paths = [p for film in sorted(films) for p in films[film]]
        vectors = np.empty((len(paths), backbone(False).num_features), dtype=np.float16)
        pool = ThreadPoolExecutor(8)
        batches = [paths[s:s + 256] for s in range(0, len(paths), 256)]
        loaded = pool.map(lambda b: [load(p) for p in b], batches)
        for n, (s, batch) in enumerate(zip(range(0, len(paths), 256), loaded)):
            vectors[s:s + len(batch)] = embed(batch)
            if n % 200 == 0:
                print(f"  {s + len(batch):,}/{len(paths):,} frames, {time.monotonic() - started:.0f}s", flush=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", vectors)
        (index_dir / "frames.json").write_text(json.dumps(frames))
        volume.commit()
        print(f"embedded {len(paths):,} frames in {time.monotonic() - started:.0f}s", flush=True)
    film_names = sorted({f for f, _ in frames})
    film_of = np.searchsorted(np.array(film_names), np.array([f for f, _ in frames]))  # frames are by film
    spans = np.searchsorted(film_of, np.arange(len(film_names) + 1))
    bank = torch.from_numpy(vectors).to(device)
    film_t = torch.from_numpy(film_of).to(device)
    out = {}
    ids = list(queries)
    for s in range(0, len(ids), 64):
        chunk = ids[s:s + 64]
        q = torch.from_numpy(embed([load(io.BytesIO(queries[i])) for i in chunk])).to(device)
        sims = (q @ bank.T).float()  # [q, frames]
        best = torch.full((len(chunk), len(film_names)), -2.0, device=device)
        best.scatter_reduce_(1, film_t.expand(len(chunk), -1), sims, reduce="amax")
        sims = sims.cpu().numpy()
        best = best.cpu().numpy()
        for row, qid in enumerate(chunk):
            rows = []
            for f in np.argsort(-best[row])[:top]:  # each film with its best frame
                a, b = spans[f], spans[f + 1]
                rows.append((film_names[f], float(best[row, f]), frames[a + int(sims[row, a:b].argmax())][1]))
            out[qid] = rows
    return out


BACKBONES = {"small": "vit_small_patch14_dinov2.lvd142m", "base": "vit_base_patch14_dinov2.lvd142m",
             "large": "vit_large_patch14_dinov2.lvd142m"}


CHUNK = 100  # films a saved part


@app.function(image=scene_image, gpu="A100-80GB", cpu=24, memory=65536, timeout=60 * 60, volumes={"/vol": volume})
def embed_dense(stride: int, sizes: list[str], limit: int = 0) -> dict:
    """Every stride-th frame of every film (indexed and unindexed footage,
    validation films too), decoded once and embedded by each DINOv2 size
    (pretrained) to scene_index/dense-<stride>/<size>/vectors.npy, with one
    frames.json ([film, frame file] a row, grouped by film) beside them."""
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from curtain_ml.train_modal import SOURCES, UNINDEXED

    started = time.monotonic()
    volume.reload()
    paths = sorted(Path(f"/vol/{SOURCES}").glob("*/*.tar"))
    indexed = {p.stem for p in paths}
    paths += sorted(p for p in Path(f"/vol/{UNINDEXED}").glob("*.tar") if p.stem not in indexed)
    paths.sort(key=lambda p: p.stem)
    if limit:  # a smoke test
        paths = paths[:limit]
    device = torch.device("cuda")
    models = {s: timm.create_model(BACKBONES[s], pretrained=True, num_classes=0, img_size=SIZE).to(device).eval().half()
              for s in sizes}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def one(path: Path) -> tuple[str, list[str], np.ndarray]:
        with tarfile.open(fileobj=io.BytesIO(path.read_bytes())) as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: frame_number(m.name))
            keep = members[stride // 2::stride]
            images = np.empty((len(keep), SIZE, SIZE, 3), dtype=np.uint8)
            for k, m in enumerate(keep):
                with Image.open(io.BytesIO(tar.extractfile(m).read())) as im:
                    im.draft("RGB", (SIZE * 2, SIZE * 2))
                    images[k] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR,
                                                                     reducing_gap=2.0))
        return path.stem, [Path(m.name).name for m in keep], images

    root = Path(f"/vol/scene_index/dense-{stride}" + (f"-test{limit}" if limit else ""))
    chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]

    def done(c: int) -> bool:
        return all((root / "parts" / f"{s}-{c:03d}.npy").exists() for s in sizes)

    todo = [p for c, chunk in enumerate(chunks) if not done(c) for p in chunk]
    print(f"{len(paths) - len(todo)} films already embedded, {len(todo)} to go", flush=True)

    def bounded(pool, ahead: int = 24):
        # pool.map would decode every film before the GPU catches up (~300 MB each)
        pending = [pool.submit(one, p) for p in todo[:ahead]]
        for k in range(len(todo)):
            if k + ahead < len(todo):
                pending.append(pool.submit(one, todo[k + ahead]))
            yield pending[k].result()
            pending[k] = None

    (root / "parts").mkdir(parents=True, exist_ok=True)
    frames, parts, waited, n = [], {s: [] for s in sizes}, 0.0, 0
    with ThreadPoolExecutor(20) as pool, torch.inference_mode():
        stream = bounded(pool)
        for c, chunk in enumerate(chunks):
            if done(c):
                continue
            for _ in chunk:
                t0 = time.monotonic()
                film, names, images = next(stream)
                waited += time.monotonic() - t0
                frames += [(film, name) for name in names]
                for s0 in range(0, len(images), 512):
                    x = torch.from_numpy(images[s0:s0 + 512]).to(device).permute(0, 3, 1, 2).half() / 255
                    x = (x - mean.half()) / std.half()
                    for s, model in models.items():
                        parts[s].append(F.normalize(model(x).float(), dim=1).cpu().numpy().astype(np.float16))
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)} films, {len(frames):,} frames, {time.monotonic() - started:.0f}s "
                          f"({waited:.0f}s waiting for decoding)", flush=True)
            for s in sizes:  # this chunk's vectors survive a timeout
                np.save(root / "parts" / f"{s}-{c:03d}.npy", np.concatenate(parts[s]))
            (root / "parts" / f"frames-{c:03d}.json").write_text(json.dumps(frames))
            volume.commit()
            frames, parts = [], {s: [] for s in sizes}
    frames = [row for c in range(len(chunks)) for row in json.loads((root / "parts" / f"frames-{c:03d}.json").read_text())]
    for s in sizes:
        (root / s).mkdir(parents=True, exist_ok=True)
        np.save(root / s / "vectors.npy", np.concatenate([np.load(root / "parts" / f"{s}-{c:03d}.npy")
                                                          for c in range(len(chunks))]))
        (root / s / "frames.json").write_text(json.dumps(frames))
    volume.commit()
    return {"films": len(paths), "frames": len(frames), "seconds": time.monotonic() - started}


@app.local_entrypoint()
def dense(stride: int = 8, sizes: str = "small,base", max_dollars: float = 6.0, limit: int = 0) -> None:
    timeout = int(max_dollars / TRAIN_RATE * 3600)
    print(f"dense-{stride} {sizes}: capped at ${max_dollars:.2f} ({timeout / 3600:.2f} h of A100)")
    print(embed_dense.with_options(timeout=timeout, retries=0).remote(stride, sizes.split(","), limit))


@app.local_entrypoint()
def prepare() -> None:
    print(prepare_all.remote())


@app.local_entrypoint()
def train(name: str, epochs: int = 1, batch_pairs: int = 512, learning_rate: float = 3e-5,
          temperature: float = 0.07, frozen_blocks: int = 6, max_dollars: float = 5.0) -> None:
    timeout = int(max_dollars / TRAIN_RATE * 3600)
    print(f"{name}: capped at ${max_dollars:.2f} ({timeout / 3600:.2f} h of A100)")
    options = {"epochs": epochs, "batch_pairs": batch_pairs, "learning_rate": learning_rate,
               "temperature": temperature, "frozen_blocks": frozen_blocks}
    print(train_scene.with_options(timeout=timeout, retries=0).remote(name, options))


@app.local_entrypoint()
def evaluate(model: str = "pretrained", max_dollars: float = 1.5) -> None:
    """Rank films for the web-search-v2 misses and every online-v1 query; writes
    .local/benchmarks/scene/<model>.json."""
    queries = {}
    for p in sorted((ROOT / ".local/benchmarks/web-search-v2/misses-site/img").glob("*.q.jpg")):
        queries["v2/" + p.name[:-len(".q.jpg")]] = p.read_bytes()
    for p in sorted((ROOT / ".local/benchmarks/online-v1/queries").glob("*.jpg")):
        queries["v1/" + p.stem] = p.read_bytes()
    timeout = int(max_dollars / EMBED_RATE * 3600)
    print(f"{len(queries)} queries; capped at ${max_dollars:.2f} ({timeout / 3600:.2f} h of L4)")
    result = embed_and_rank.with_options(timeout=timeout, retries=0).remote(model, queries)
    target = ROOT / ".local/benchmarks/scene" / f"{model}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result))
    print(f"wrote {target}")
