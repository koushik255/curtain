"""Train a small composition-similarity head on frozen DINOv2 features (CPU-friendly).

Idea: the pilot's similarity was dominated by tone (dark, monochrome) and
subject (a face). We teach a small head to ignore those by showing it each
frame under several appearance changes (grayscale, contrast/gamma, blur,
recolouring, a slight reframing) and training it so all versions of one frame
map together while different frames stay apart (InfoNCE). What survives is
mostly layout: where masses, figures and depth sit in a 4 x 4 grid.

Steps (outputs in results/composition-v1/, pilot untouched):
    ../.venv/bin/python composition_model.py extract   # backbone features, cached per film
    ../.venv/bin/python composition_model.py train     # trains the head (minutes on CPU)
    ../.venv/bin/python composition_model.py search    # ranks Kurosawa shots vs focus films
Optional later: --labels kurosawa-review.json (gallery export) for fine-tuning.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

import kurosawa_experiment as pilot
from focused_experiment import TARGETS

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/composition-v1"
CACHE = HERE / "results/composition-cache"
FEATURE_VERSION = "dinov2-s14-grid4x4-views3-v1"
VIEWS = 3  # 0 = original, 1-2 = appearance-scrambled
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


# ---------------------------------------------------------------- augmentation
def scramble(rgb: Image.Image, rng: random.Random) -> Image.Image:
    """Destroy tone/colour/texture cues while keeping the layout roughly intact."""
    w, h = rgb.size
    s = rng.uniform(.88, 1.0)  # slight reframing: composition should tolerate it
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = rgb.crop((x, y, x + cw, y + ch))
    arr = np.asarray(im, dtype=np.float32) / 255
    if rng.random() < .5:
        arr = arr.mean(axis=2, keepdims=True).repeat(3, axis=2)  # grayscale
    else:
        arr = arr[..., rng.sample(range(3), 3)]  # channel shuffle
        arr = arr * np.array([rng.uniform(.6, 1.4) for _ in range(3)])  # tint
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-3), 0, 1)  # normalise exposure
    arr = arr ** rng.uniform(.5, 2.0)  # gamma: dark <-> bright
    arr = arr + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, .05), arr.shape)
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2.5)))


def to_tensor(im: Image.Image) -> torch.Tensor:
    padded = ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    x = torch.from_numpy(np.asarray(padded).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def load_views(record, views):
    with Image.open(record["path"]) as image:
        rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    if float(gray.std()) < 8 or float(gray.mean()) < 8:
        return None
    rng = random.Random(record["filename"])  # deterministic per frame
    return [to_tensor(rgb)] + [to_tensor(scramble(rgb, rng)) for _ in range(views - 1)]


# ---------------------------------------------------------------- extraction
def extract_movie(model, records, views, pool, batch=32):
    key = hashlib.sha256(json.dumps({"v": FEATURE_VERSION, "views": views,
                                     "frames": [r["filename"] for r in records]}).encode()).hexdigest()
    path = CACHE / f"{key}.npz"
    if path.exists():
        with np.load(path) as d:
            return [records[i] for i in d["keep"]], d["grid"].copy(), True
    grids, keep = [], []
    with torch.inference_mode():
        for start in range(0, len(records), batch):
            loaded = list(pool.map(lambda r: load_views(r, views), records[start:start + batch]))
            valid = [(start + i, v) for i, v in enumerate(loaded) if v is not None]
            if not valid:
                continue
            x = torch.stack([t for _, v in valid for t in v])
            p = model.forward_features(x)["x_norm_patchtokens"].float()
            g = p.reshape(-1, 4, 4, 4, 4, 384).mean(dim=(2, 4)).reshape(len(valid), views, 16, 384)
            grids.append(g.half().numpy()); keep.extend(i for i, _ in valid)
    grid = np.concatenate(grids) if grids else np.zeros((0, views, 16, 384), np.float16)
    tmp = path.with_suffix(".partial.npz")
    np.savez(tmp, keep=np.asarray(keep, dtype=int), grid=grid)
    tmp.replace(path)
    return [records[i] for i in keep], grid, False


def build_sets(root):
    catalog = json.loads((root / ".local/faiss-all-films-v1/catalog.json").read_text())
    train, search = [], []
    for m in catalog["movies"]:
        stem = m["movie"].split("/")[-1]
        if stem in pilot.EXCLUDED:
            continue
        k = stem in pilot.KUROSAWA
        # Training: same sparse sample as the pilot (all films). Search: every shot
        # of Kurosawa and focus films.
        train.append(pilot.select_frames(root, catalog, m, 96 if k else 16))
        if k or stem in TARGETS:
            search.append(pilot.select_frames(root, catalog, m, 100_000))
    return train, search


def extract(args):
    torch.set_num_threads(args.threads)
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    ckpt = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    model = pilot.load_model(hub, ckpt)
    train, search = build_sets(args.root)
    for name, sets, views in (("train", train, VIEWS), ("search", search, 1)):
        recs, grids = [], []
        with ThreadPoolExecutor(6) as pool:
            for n, sel in enumerate(sets, 1):
                kept, g, cached = extract_movie(model, sel, views, pool)
                recs.extend(kept); grids.append(g)
                if n % 25 == 0 or n == len(sets) or not cached:
                    pilot.note(f"{name} {n}/{len(sets)} {sel[0]['title']}: {len(kept)}" + (" (cached)" if cached else ""))
        np.save(OUT / f"{name}-grid.npy", np.concatenate(grids))
        pilot.write_json(OUT / f"{name}-records.json", recs)


def shot_neighbours(args):
    """v2: two extra frames from the same shot (near its start and end).

    Within a shot, actors and camera move but the staging mostly holds, so these
    are harder positives than appearance scrambles of one frame. View 3 is the
    early frame, scrambled; view 4 is the late frame, original colours.
    Aligned row-for-row with train-records.json.
    """
    torch.set_num_threads(args.threads)
    catalog = json.loads((args.root / ".local/faiss-all-films-v1/catalog.json").read_text())
    first = {m["movie"]: m["first"] for m in catalog["movies"]}
    records = json.loads((OUT / "train-records.json").read_text())
    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    ckpt = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    model = pilot.load_model(hub, ckpt)

    def frames_for(r):
        lo, hi = r["shot_start_row"], r["shot_end_row"]
        a, b = lo + (hi - lo) // 6, hi - 1 - (hi - lo) // 6
        out = []
        for row in (a, b):
            name = catalog["names"][first[r["movie"]] + row]
            out.append(str(pilot.frame_path(args.root, name)))
        return out

    def load(r):
        pa, pb = frames_for(r)
        rng = random.Random("shot:" + r["filename"])
        with Image.open(pa) as im:
            va = to_tensor(scramble(im.convert("RGB"), rng))
        with Image.open(pb) as im:
            vb = to_tensor(im.convert("RGB"))
        return [va, vb]

    by_movie = {}
    for i, r in enumerate(records):
        by_movie.setdefault(r["movie"], []).append(i)
    result = np.zeros((len(records), 2, 16, 384), np.float16)
    with ThreadPoolExecutor(6) as pool, torch.inference_mode():
        for n, (movie, idx) in enumerate(by_movie.items(), 1):
            key = hashlib.sha256(json.dumps({"v": FEATURE_VERSION + "-shotpair", "frames": [records[i]["filename"] for i in idx]}).encode()).hexdigest()
            path = CACHE / f"{key}.npy"
            if path.exists():
                result[idx] = np.load(path); continue
            grids = []
            for s in range(0, len(idx), 32):
                views = list(pool.map(load, [records[i] for i in idx[s:s + 32]]))
                p = model.forward_features(torch.stack([t for v in views for t in v]))["x_norm_patchtokens"].float()
                grids.append(p.reshape(-1, 4, 4, 4, 4, 384).mean(dim=(2, 4)).reshape(len(views), 2, 16, 384).half().numpy())
            g = np.concatenate(grids); np.save(path, g); result[idx] = g
            if n % 25 == 0 or n == len(by_movie):
                pilot.note(f"shot pairs {n}/{len(by_movie)}")
    np.save(OUT / "train-shot-grid.npy", result)


# ---------------------------------------------------------------- model
class CompositionHead(nn.Module):
    """Per-cell projection + learned position + tiny transformer -> 128-d vector."""

    def __init__(self, dim=384, width=128, out=128):
        super().__init__()
        self.cell = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.pos = nn.Parameter(torch.randn(16, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, 2)
        self.proj = nn.Sequential(nn.LayerNorm(16 * width), nn.Linear(16 * width, out))

    def forward(self, grid):  # (B, 16, 384)
        h = self.mix(self.cell(grid) + self.pos)
        return F.normalize(self.proj(h.flatten(1)), dim=-1)


def train(args):
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    grid = torch.from_numpy(np.load(OUT / "train-grid.npy").astype(np.float32))  # (N, V, 16, 384)
    records = json.loads((OUT / "train-records.json").read_text())
    films = np.array([r["movie"] for r in records])
    # Hold out ~10% of films to watch for overfitting.
    names = sorted(set(films)); random.Random(0).shuffle(names)
    held = set(names[: len(names) // 10])
    tr = np.where(~np.isin(films, list(held)))[0]; va = np.where(np.isin(films, list(held)))[0]
    head = CompositionHead()
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.05)
    temp = .07

    def loss_on(idx):
        v = torch.randint(0, VIEWS, (2,))
        while v[0] == v[1]:
            v = torch.randint(0, VIEWS, (2,))
        a, b = head(grid[idx, v[0]]), head(grid[idx, v[1]])
        logits = a @ b.T / temp
        y = torch.arange(len(idx))
        acc = (logits.argmax(1) == y).float().mean().item()
        return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2, acc

    log = []
    for epoch in range(args.epochs):
        head.train(); perm = np.random.default_rng(epoch).permutation(tr)
        tl = []
        for s in range(0, len(perm) - args.batch + 1, args.batch):
            loss, _ = loss_on(torch.from_numpy(perm[s:s + args.batch]))
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        head.eval()
        with torch.no_grad():
            vl, vacc = loss_on(torch.from_numpy(va[:args.batch]))
        log.append(dict(epoch=epoch, train_loss=float(np.mean(tl)), val_loss=vl.item(), val_top1=vacc))
        pilot.note(f"epoch {epoch}: train {np.mean(tl):.3f}  val {vl.item():.3f}  val top-1 {vacc:.2f}")
    torch.save(head.state_dict(), OUT / "head.pt")
    pilot.write_json(OUT / "train-log.json", dict(held_out_films=sorted(held), log=log))


# ---------------------------------------------------------------- search
def search(args):
    torch.set_num_threads(args.threads)
    head = CompositionHead(); head.load_state_dict(torch.load(OUT / "head.pt")); head.eval()
    grid = torch.from_numpy(np.load(OUT / "search-grid.npy")[:, 0].astype(np.float32))
    records = json.loads((OUT / "search-records.json").read_text())
    with torch.no_grad():
        z = torch.cat([head(grid[s:s + 1024]) for s in range(0, len(grid), 1024)])
    # Hub penalty: frames that resemble everything (dark close-ups) are down-weighted
    # by subtracting their mean similarity to a random background sample (CSLS-style).
    bg = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(0))[:2000]]
    hub = (z @ bg.T).topk(20, dim=1).values.mean(1)
    src = [i for i, r in enumerate(records) if r["kurosawa"]]
    out = {}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    for stem in TARGETS:
        tgt = [i for i, r in enumerate(records) if r["movie"].endswith("/" + stem)]
        raw = z[src] @ z[tgt].T
        score = (2 * raw - hub[src][:, None] - hub[tgt][None, :]).numpy()
        picked, used_s, used_t, per = [], [], [], {}
        for k in np.argsort(score, axis=None)[::-1][:200_000]:
            a, b = divmod(int(k), len(tgt)); s, t = src[a], tgt[b]
            sr, trr = records[s], records[t]
            if per.get(sr["movie"], 0) >= 3: continue
            if any(records[x]["movie"] == sr["movie"] and abs(records[x]["shot"] - sr["shot"]) < 3 for x in used_s): continue
            if any(abs(records[x]["shot"] - trr["shot"]) < 3 for x in used_t): continue
            picked.append(dict(source=sr["filename"], source_time=sr["time"], target=trr["filename"],
                               target_time=trr["time"], score=round(float(score[a, b]), 4),
                               cosine=round(float(raw[a, b]), 4)))
            used_s.append(s); used_t.append(t); per[sr["movie"]] = per.get(sr["movie"], 0) + 1
            if len(picked) == 12: break
        out[stem] = picked
        canvas = Image.new("RGB", (1000, 50 + 230 * len(picked)), "#171b22"); d = ImageDraw.Draw(canvas)
        d.text((15, 15), f"{stem} — trained composition head (composition-v1)", fill="white", font=font)
        by_name = {r["filename"]: r for r in records}
        for n, pr in enumerate(picked):
            y = 50 + 230 * n
            for col, key in enumerate(("source", "target")):
                r = by_name[pr[key]]
                with Image.open(r["path"]) as im:
                    th = ImageOps.contain(im.convert("RGB"), (470, 185))
                canvas.paste(th, (15 + 500 * col + (470 - th.width) // 2, y))
                d.text((15 + 500 * col, y + 188), f"{r['title'][:34]} ({r['year']}) ~{r['time']} shot {r['shot']}", fill="white", font=font)
            d.text((15, y + 206), f"#{n} score {pr['score']:.3f} cosine {pr['cosine']:.3f}", fill="#96a9bd", font=font)
        canvas.save(OUT / f"sheet-{stem}.jpg", quality=88)
    pilot.write_json(OUT / "search.json", out)
    pilot.note(f"Search complete: {OUT}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["extract", "shots", "train", "search"])
    p.add_argument("--root", type=Path, default=HERE.parent)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()
    dict(extract=extract, shots=shot_neighbours, train=train, search=search)[args.step](args)


if __name__ == "__main__":
    main()
