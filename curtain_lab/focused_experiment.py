"""Focused dense follow-up to the Kurosawa pilot (pilot outputs are untouched).

Pilot review found the main failure was coverage: with 16 frames per target
film, the same target frame was often the best match for several Kurosawa
shots (e.g. one Star Wars frame topped three different source films). This run
keeps the pilot's scorer unchanged (same DINOv2 features, same 50/50 blend) and
changes only sampling: every shot midpoint in the eleven Kurosawa films and in
a small list of target films chosen for a possible, separately sourceable
Kurosawa connection or a period-genre control.

    ../.venv/bin/python focused_experiment.py

Outputs go to results/focused-dense-v1/. Embeddings share results/embedding-cache/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

import kurosawa_experiment as pilot

# Why each title is here. These are reasons to look, not influence claims.
TARGETS = {
    "For_a_Few_Dollars_More_1965": "Leone; follows A Fistful of Dollars, the unauthorized Yojimbo remake",
    "Once_Upon_a_Time_in_the_West_1968": "Leone",
    "Star_Wars_1977": "Lucas has discussed Kurosawa (Hidden Fortress is not available here)",
    "Unforgiven_1992": "Eastwood, star of A Fistful of Dollars",
    "Kill_Bill_Vol_1_2003": "Tarantino; samurai-film homage",
    "Apocalypse_Now_1979": "Coppola, executive producer of Kagemusha's international release",
    "Harakiri_1962": "Contemporary jidaigeki control (Kobayashi, same year as Sanjuro)",
    "The_Sword_of_Doom_1966": "Contemporary jidaigeki control (Okamoto, Mifune appears)",
}
SHOTS = 100_000  # effectively every eligible shot midpoint
PER_TARGET = 12
PER_SOURCE_FILM = 3
MIN_SHOT_GAP = 3  # treat neighbouring shots as the same moment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=here.parent)
    parser.add_argument("--out", type=Path, default=here / "results/focused-dense-v1")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "embedding-cache"; cache.mkdir(exist_ok=True)
    catalog = json.loads((args.root / ".local/faiss-all-films-v1/catalog.json").read_text())
    wanted = pilot.KUROSAWA | set(TARGETS)
    movies = [m for m in catalog["movies"] if m["movie"].split("/")[-1] in wanted]
    found = {m["movie"].split("/")[-1] for m in movies}
    if found != wanted:
        raise ValueError(f"Missing titles: {sorted(wanted - found)}")
    sampled = [pilot.select_frames(args.root, catalog, m, SHOTS) for m in movies]
    del catalog
    sampled.sort(key=lambda rs: (not rs[0]["kurosawa"], rs[0]["movie"]))
    pilot.note(f"{len(sampled)} films, {sum(map(len, sampled)):,} shot midpoints")

    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != pilot.CHECKPOINT_SHA256:
        raise ValueError("Model checkpoint changed")
    model = pilot.load_model(hub, checkpoint)
    records, gs, ps = [], [], []
    with ThreadPoolExecutor(4) as pool:
        for n, selected in enumerate(sampled, 1):
            kept, g, p, cached = pilot.embed_movie(model, selected, cache, args.batch_size, pool)
            records.extend(kept); gs.append(g); ps.append(p)
            pilot.note(f"{n}/{len(sampled)} {selected[0]['title']}: {len(kept)} frames" + (" (cached)" if cached else ""))
    g = F.normalize(torch.from_numpy(np.concatenate(gs).astype(np.float32)), dim=1)
    p = F.normalize(torch.from_numpy(np.concatenate(ps).astype(np.float32)), dim=1)
    source = [i for i, r in enumerate(records) if r["kurosawa"]]

    per_target = {}
    for stem, reason in TARGETS.items():
        target = [i for i, r in enumerate(records) if r["movie"].endswith("/" + stem)]
        gsc = (g[source] @ g[target].T).numpy(); psc = (p[source] @ p[target].T).numpy()
        score = .5 * gsc + .5 * psc
        year = records[target[0]]["year"]
        flat = np.argsort(score, axis=None)[::-1]
        picked, used_s, used_t, per_film = [], [], [], {}
        for k in flat[:200_000]:
            a, b = divmod(int(k), len(target))
            s, t = source[a], target[b]
            sr, tr = records[s], records[t]
            if per_film.get(sr["movie"], 0) >= PER_SOURCE_FILM:
                continue
            if any(records[x]["movie"] == sr["movie"] and abs(records[x]["shot"] - sr["shot"]) < MIN_SHOT_GAP for x in used_s):
                continue
            if any(abs(records[x]["shot"] - tr["shot"]) < MIN_SHOT_GAP for x in used_t):
                continue
            picked.append(dict(source=s, target=t, score=round(float(score[a, b]), 5),
                               global_score=round(float(gsc[a, b]), 5),
                               spatial_score=round(float(psc[a, b]), 5),
                               chronology="later" if year > sr["year"] else "earlier" if year < sr["year"] else "same year"))
            used_s.append(s); used_t.append(t)
            per_film[sr["movie"]] = per_film.get(sr["movie"], 0) + 1
            if len(picked) == PER_TARGET:
                break
        per_target[stem] = dict(reason=reason, frames=len(target), pairs=picked)

    used = {x for v in per_target.values() for pr in v["pairs"] for x in (pr["source"], pr["target"])}
    pilot.make_assets(out, records, used)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    for stem, v in per_target.items():
        rows = v["pairs"]
        canvas = Image.new("RGB", (1000, 50 + 230 * len(rows)), "#171b22")
        d = ImageDraw.Draw(canvas)
        d.text((15, 15), f"{stem} — dense shot search. Resemblance only; not evidence of influence.", fill="white", font=font)
        for n, pr in enumerate(rows):
            y = 50 + 230 * n
            for col, i in enumerate((pr["source"], pr["target"])):
                r = records[i]
                with Image.open(out / r["asset"]) as im:
                    th = ImageOps.contain(im, (470, 185))
                canvas.paste(th, (15 + 500 * col + (470 - th.width) // 2, y))
                d.text((15 + 500 * col, y + 188), f"{r['title'][:34]} ({r['year']}) ~{r['time']} shot {r['shot']}", fill="white", font=font)
            d.text((15, y + 206), f"#{n} score {pr['score']:.3f}  global {pr['global_score']:.2f}  spatial {pr['spatial_score']:.2f}  {pr['chronology']}", fill="#96a9bd", font=font)
        canvas.save(out / f"sheet-{stem}.jpg", quality=88)

    keep = sorted(used)
    summary = dict(generated=datetime.now(timezone.utc).isoformat(), version=pilot.VERSION,
                   change_from_pilot="every shot midpoint for sources and the listed targets; scorer unchanged",
                   frames={r["title"]: sum(x["title"] == r["title"] for x in records) for r in records},
                   targets=TARGETS)
    pilot.write_json(out / "results.json", dict(summary=summary,
                     records={i: records[i] for i in keep}, per_target=per_target))
    pilot.note(f"Complete: {out}")


if __name__ == "__main__":
    main()
