"""Offline, resumable Kurosawa shot-similarity pilot using Curtain's existing frames.

Run with ../.venv/bin/python kurosawa_experiment.py. Outputs stay in this lab.
No downloads, training, live-site changes, or writes to Curtain's source data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F
from rcmindex.data_paths import ORIGINAL_DIRS, WEBP


KUROSAWA = {
    "Dreams_1990", "High_and_Low_1963", "Ikiru_1952", "Kagemusha_1980", "Ran_1985",
    "Red_Beard_1965",
    "Rashomon_1950", "Sanjuro_1962", "Seven_Samurai_1954",
    "Throne_of_Blood_1957", "Yojimbo_1961",
}
EXCLUDED = {
    "The_Hidden_Fortress_1958": "Mislabeled source: inspected frames contain interviews and film excerpts, consistent with a documentary extra rather than the feature. Excluded from both source and target sets."
}
VERSION = "dinov2-s14-rgb-pad224-cls-and-spatial2x2-v1"
CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
START = time.monotonic()


def note(message):
    print(f"[{time.monotonic() - START:6.1f}s] {message}", flush=True)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def clock(seconds):
    seconds = round(seconds)
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def frame_path(root, name):
    relative = Path(name)
    small = WEBP / relative.with_suffix(".webp")
    if small.exists():
        return small
    collection, movie, filename = relative.parts
    base = ORIGINAL_DIRS[collection]
    original = base / movie / filename
    if original.exists():
        return original
    raise FileNotFoundError(name)


def select_frames(root, catalog, movie, limit):
    shot_path = root / ".local/shots" / (movie["movie"] + ".json")
    shots = json.loads(shot_path.read_text())
    if shots["frames"] != movie["frames"] or shots["fps"] != 2:
        raise ValueError(f"Shot map mismatch: {movie['movie']}")
    edges = [0, *shots["cuts"], movie["frames"]]
    if edges != sorted(set(edges)):
        raise ValueError(f"Invalid shot boundaries: {movie['movie']}")
    # Avoid the first/last 2% (often logos/credits); retain distinct shot midpoints.
    candidates = [(shot, (lo + hi - 1) // 2, lo, hi)
                  for shot, (lo, hi) in enumerate(zip(edges, edges[1:]))
                  if .02 * movie["frames"] <= (lo + hi - 1) // 2 < .98 * movie["frames"]]
    chosen = np.unique(np.linspace(0, len(candidates) - 1,
                                   min(limit, len(candidates))).round().astype(int))
    stem = movie["movie"].split("/")[-1]
    title, _, year = stem.rpartition("_")
    output = []
    for i in chosen:
        shot, row, lo, hi = candidates[int(i)]
        name = catalog["names"][movie["first"] + row]
        path = frame_path(root, name)
        number = int(path.stem.split("_")[-1])
        seconds = number / 2  # Curtain's extraction convention, not edition-verified.
        output.append(dict(movie=movie["movie"], title=title.replace("_", " "),
                           year=int(year), kurosawa=stem in KUROSAWA,
                           shot=shot, row=row, shot_start_row=lo, shot_end_row=hi,
                           filename=name, path=str(path), seconds=seconds,
                           time=clock(seconds), timestamp_kind="approximate_2fps_frame_clock"))
    return output


def load_image(record):
    with Image.open(record["path"]) as image:
        rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    # Reject near-empty frames, not a full titles/credits detector.
    if float(gray.std()) < 8 or float(gray.mean()) < 8:
        return None
    padded = ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BICUBIC,
                          color=(0, 0, 0))
    x = torch.from_numpy(np.asarray(padded).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def load_model(hub, checkpoint):
    model = torch.hub.load(str(hub), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def embed_movie(model, records, cache, batch_size, pool):
    signature = hashlib.sha256(json.dumps({
        "version": VERSION, "frames": [r["filename"] for r in records],
        "files": [(Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns)
                  for r in records],
    }, sort_keys=True).encode()).hexdigest()
    npz = cache / f"{signature}.npz"
    if npz.exists():
        with np.load(npz) as data:
            keep = data["keep"].tolist()
            return [records[i] for i in keep], data["global"].copy(), data["spatial"].copy(), True
    globals_, spatial_, keep = [], [], []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            loaded = list(pool.map(load_image, records[start:start + batch_size]))
            valid = [(start + i, x) for i, x in enumerate(loaded) if x is not None]
            if not valid:
                continue
            features = model.forward_features(torch.stack([x for _, x in valid]))
            cls = F.normalize(features["x_norm_clstoken"].float(), dim=-1)
            patches = features["x_norm_patchtokens"].float().reshape(-1, 16, 16, 384)
            # Pool four fixed quadrants: retain coarse spatial arrangement.
            cells = patches.reshape(-1, 2, 8, 2, 8, 384).mean(dim=(2, 4)).reshape(-1, 4, 384)
            cells = F.normalize(cells, dim=-1).reshape(-1, 1536) / 2
            globals_.append(cls.numpy()); spatial_.append(cells.numpy())
            keep.extend(i for i, _ in valid)
    if keep:
        global_v, spatial_v = np.concatenate(globals_), np.concatenate(spatial_)
    else:
        global_v, spatial_v = np.empty((0, 384), np.float32), np.empty((0, 1536), np.float32)
    temporary = npz.with_suffix(".partial.npz")
    np.savez_compressed(temporary, keep=np.asarray(keep), **{
        "global": global_v.astype(np.float16), "spatial": spatial_v.astype(np.float16)})
    temporary.replace(npz)
    return [records[i] for i in keep], global_v, spatial_v, False


def rank(records, global_v, spatial_v, topk):
    source = [i for i, r in enumerate(records) if r["kurosawa"]]
    target = [i for i, r in enumerate(records) if not r["kurosawa"]]
    g = torch.from_numpy(global_v.astype(np.float32))
    p = torch.from_numpy(spatial_v.astype(np.float32))
    # Re-normalize after fp16 cache roundtrips, preserving unit-length scoring.
    g = F.normalize(g, dim=1); p = F.normalize(p, dim=1)
    tg, tp = g[target], p[target]
    queries, pairs, by_film = [], [], {}
    for start in range(0, len(source), 64):
        ids = source[start:start + 64]
        gs = (g[ids] @ tg.T).numpy()
        ps = (p[ids] @ tp.T).numpy()
        for row, q in enumerate(ids):
            score = .5 * gs[row] + .5 * ps[row]
            query = {"source": q}
            # Earlier films form a comparison group; never label them downstream influence.
            for mode in ("later", "earlier"):
                eligible = np.asarray([
                    r["year"] > records[q]["year"] if mode == "later"
                    else r["year"] < records[q]["year"]
                    for r in (records[t] for t in target)])
                order = np.argsort(np.where(eligible, score, -np.inf))[::-1]
                seen, matches = set(), []
                for j in order:
                    if not eligible[j]:
                        break
                    t = target[j]
                    if records[t]["movie"] in seen:
                        continue
                    seen.add(records[t]["movie"])
                    match = dict(source=q, target=t, score=round(float(score[j]), 5),
                                 global_score=round(float(gs[row, j]), 5),
                                 spatial_score=round(float(ps[row, j]), 5))
                    if mode == "later":
                        # Keep the strongest pair per source film for each target film,
                        # so a specific-film lookup is possible even outside the top five.
                        options = by_film.setdefault(records[t]["movie"], {})
                        source_movie = records[q]["movie"]
                        if source_movie not in options or match["score"] > options[source_movie]["score"]:
                            options[source_movie] = match
                    if len(matches) < topk:
                        matches.append(match)
                        if mode == "later":
                            pairs.append(match)
                    if mode == "earlier" and len(matches) == topk:
                        break
                query[mode] = matches
            queries.append(query)
    # Curate by score and diversity only, never call this human validation.
    shortlist, used_source, used_target = [], set(), set()
    film_counts, source_counts = {}, {}
    for pair in sorted(pairs, key=lambda x: -x["score"]):
        q, t = pair["source"], pair["target"]
        movie, source_movie = records[t]["movie"], records[q]["movie"]
        if (q in used_source or t in used_target or film_counts.get(movie, 0) >= 3
                or source_counts.get(source_movie, 0) >= 12):
            continue
        shortlist.append(pair); used_source.add(q); used_target.add(t)
        film_counts[movie] = film_counts.get(movie, 0) + 1
        source_counts[source_movie] = source_counts.get(source_movie, 0) + 1
        if len(shortlist) >= 100:
            break
    film_matches = [pair for options in by_film.values()
                    for pair in sorted(options.values(), key=lambda p: -p["score"])[:3]]
    return queries, shortlist, film_matches


def make_assets(out, records, used):
    assets = out / "frames"
    assets.mkdir(exist_ok=True)
    def one(i):
        r = records[i]
        dest = assets / (hashlib.sha256(r["filename"].encode()).hexdigest()[:20] + ".jpg")
        if not dest.exists():
            with Image.open(r["path"]) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail((640, 480))
                rgb.save(dest, quality=85)
        return i, str(dest.relative_to(out))
    with ThreadPoolExecutor(6) as pool:
        for i, relative in pool.map(one, sorted(used)):
            records[i]["asset"] = relative


def contact_sheet(out, records, shortlist):
    pairs = shortlist[:12]
    width, row_h = 1000, 240
    canvas = Image.new("RGB", (width, 85 + row_h * len(pairs)), "#171b22")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((18, 12), "Kurosawa / visual similarity pilot", fill="white", font=font)
    draw.text((18, 40), "Automatically ranked leads. Visual resemblance does not establish influence.", fill="#b9c6d4", font=font)
    for n, pair in enumerate(pairs):
        y = 85 + row_h * n
        for col, i in enumerate((pair["source"], pair["target"])):
            r = records[i]
            with Image.open(out / r["asset"]) as im:
                thumb = ImageOps.contain(im, (470, 190))
                canvas.paste(thumb, (15 + 500 * col + (470 - thumb.width) // 2, y))
            title = r['title']
            label = f"{title} ({r['year']}) ~{r['time']}"
            while draw.textlength(label, font=font) > 470 and len(title) > 4:
                title = title[:-2]
                label = f"{title}… ({r['year']}) ~{r['time']}"
            draw.text((15 + 500 * col, y + 195), label, fill="white", font=font)
        draw.text((15, y + 218), f"Similarity {pair['score']:.3f} (ranking score)", fill="#96a9bd", font=font)
    canvas.save(out / "contact-sheet.jpg", quality=90)


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kurosawa / Curtain experiment</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#10141b;color:#e8edf5}body{max-width:1250px;margin:auto;padding:32px 20px}h1{font-size:36px;margin:8px 0}p{line-height:1.6;color:#b6c3d3;max-width:950px}.eyebrow{color:#a8d5cb;letter-spacing:.13em;font-size:12px}label{display:inline-flex;flex-direction:column;gap:6px;font-size:13px;color:#b6c3d3}select,button{padding:10px;background:#1b2330;border:1px solid #435168;border-radius:7px;color:#edf3fa;font:inherit}button{cursor:pointer}nav{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.pair{border:1px solid #303d50;border-radius:12px;background:#171e29;padding:16px;margin:16px 0}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;height:230px;object-fit:contain;background:#080b10}figcaption{font-size:14px;line-height:1.6;margin-top:7px}.sub{font-size:13px;color:#97a9bf}.score{color:#a8d5cb;font-size:13px}.review{display:flex;gap:8px;margin-top:12px}.active{border-color:#8ad4bd;background:#24443d}#pager{display:flex;align-items:center;gap:16px}.empty{padding:20px}details{margin:16px 0}a{color:#a8d5cb}@media(max-width:650px){.images{grid-template-columns:1fr}img{height:200px}h1{font-size:28px}}
</style>
<div class="eyebrow">CURTAIN LAB · EXPERIMENT 01</div><h1>Kurosawa, across the frame</h1>
<p id="intro"></p><p>These are visual search leads for close viewing. Scores measure feature similarity; they are not probabilities of influence. Frame-clock times are approximate and may differ from your edition.</p>
<p id="exclusions"></p>
<details><summary>Method and limits</summary><p id="method"></p><p>Sampling skips the first and last 2% of each indexed movie, uses shot midpoints, and rejects nearly blank images. It can miss brief compositions and changes within a shot. Collection coverage is not a guarantee of complete or correctly labeled film copies. Shared subjects, costumes, monochrome photography, and aspect ratios can drive matches. The earlier-film comparison group helps expose shared conventions; it is not a statistical validation. Release year alone does not establish who saw what. Motion, editing, sound, and narrative are not analyzed.</p><p>Review marks are held in this page for this session. Export them before closing or refreshing.</p></details>
<nav><label>View<select id="view"><option value="shortlist">Diverse shortlist</option><option value="films">Best pairs per later film</option><option value="later">All references → later films</option><option value="earlier">All references → earlier films</option></select></label><label>Kurosawa film<select id="film"><option value="">All reference films</option></select></label><label>Other film<select id="target"><option value="">All other films</option></select></label><button id="export">Export review notes</button></nav>
<div id="pager"><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button></div><div id="results"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), records=data.records;
const $=id=>document.getElementById(id);let page=0;const notes={};const size=12;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('intro').textContent=`${data.summary.source_movies} Kurosawa films · ${data.summary.target_movies} other films · ${data.summary.source_frames.toLocaleString()} reference frames · ${data.summary.target_frames.toLocaleString()} comparison frames. This is a sparse first pass through the existing collection.`;
$('method').textContent=data.summary.method;
$('exclusions').textContent='Excluded: The Hidden Fortress folder contains interviews and film excerpts rather than the feature. The eleven included reference titles passed a sample-based visual identity check.';
for(const [id,isSource] of [['film',true],['target',false]]){
 const films=new Map(records.filter(r=>r.kurosawa===isSource).map(r=>[r.movie,`${r.title} (${r.year})`]));
 for(const [value,label] of [...films].sort((a,b)=>a[1].localeCompare(b[1]))){const o=document.createElement('option');o.value=value;o.textContent=label;$(id).append(o);}
}
function key(p){return `${p.source}:${p.target}`;}
function figure(i){const r=records[i];return `<figure><a href="${esc(r.asset)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(r.asset)}" alt="Frame from ${esc(r.title)}"></a><figcaption>${esc(r.title)} (${r.year})<div class="sub">~${r.time} · shot ${r.shot+1} · ${esc(r.filename.split('/').pop())}</div></figcaption></figure>`;}
function render(){
 const mode=$('view').value;let pairs=mode==='shortlist'?data.shortlist:mode==='films'?data.film_matches:data.queries.flatMap(q=>q[mode]);
 pairs=pairs.filter(p=>(!$('film').value||records[p.source].movie===$('film').value)&&(!$('target').value||records[p.target].movie===$('target').value));
 if(mode!=='shortlist')pairs.sort((a,b)=>b.score-a.score);
 const pages=Math.max(1,Math.ceil(pairs.length/size));page=Math.max(0,Math.min(page,pages-1));
 $('count').textContent=`${pairs.length} pairs · page ${page+1} of ${pages}`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;
 $('results').innerHTML=pairs.slice(page*size,(page+1)*size).map(p=>`<article class="pair"><div class="images">${figure(p.source)}${figure(p.target)}</div><p class="score">Similarity ${p.score.toFixed(3)} · global ${p.global_score.toFixed(3)} · spatial ${p.spatial_score.toFixed(3)}</p><div class="review">${['Useful lead','Superficial','Unsure'].map(mark=>`<button class="${notes[key(p)]?.mark===mark?'active':''}" data-pair="${key(p)}" data-mark="${mark}">${mark}</button>`).join('')}</div><label style="display:block;margin-top:10px">What is shared?<input style="width:95%;padding:9px;margin-top:6px;background:#10141b;border:1px solid #435168;color:inherit" data-note="${key(p)}" value="${esc(notes[key(p)]?.note||'')}" placeholder="Composition, blocking, depth, or a superficial subject match…"></label></article>`).join('')||'<p class="empty">No retrieved pairs for these filters. Sparse sampling and top-five retrieval do not cover every film pair.</p>';
}
for(const id of ['view','film','target'])$(id).onchange=()=>{page=0;render();};
$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
$('results').onclick=e=>{const b=e.target.closest('[data-pair]');if(!b)return;notes[b.dataset.pair]={...notes[b.dataset.pair],mark:b.dataset.mark};render();};
$('results').oninput=e=>{if(e.target.dataset.note)notes[e.target.dataset.note]={...notes[e.target.dataset.note],note:e.target.value};};
$('export').onclick=()=>{const rows=Object.entries(notes).map(([k,v])=>{const [q,t]=k.split(':').map(Number);return {source:records[q],target:records[t],...v};});const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify({experiment:data.summary,review:rows},null,2)],{type:'application/json'}));a.href=url;a.download='kurosawa-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results/pilot-v1")
    parser.add_argument("--source-shots", type=int, default=96)
    parser.add_argument("--target-shots", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    if min(args.source_shots, args.target_shots, args.batch_size, args.threads, args.topk) < 1:
        parser.error("Shot counts, batch size, thread count, and topk must be positive")
    torch.set_num_threads(args.threads)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "embedding-cache"; cache.mkdir(exist_ok=True)
    catalog_path = args.root / ".local/faiss-all-films-v1/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    found = {m["movie"].split("/")[-1] for m in catalog["movies"]} & KUROSAWA
    if found != KUROSAWA:
        raise ValueError(f"Missing Kurosawa titles: {sorted(KUROSAWA - found)}")
    inventory = []
    sampled = []
    for movie in catalog["movies"]:
        stem = movie["movie"].split("/")[-1]
        if stem in EXCLUDED:
            inventory.append(dict(movie=movie["movie"], excluded=True,
                                  reason=EXCLUDED[stem], indexed_frames=movie["frames"]))
            continue
        is_source = stem in KUROSAWA
        records = select_frames(args.root, catalog, movie,
                                args.source_shots if is_source else args.target_shots)
        sampled.append(records)
        first_name = catalog["names"][movie["first"]]
        last_name = catalog["names"][movie["first"] + movie["frames"] - 1]
        inventory.append(dict(movie=movie["movie"], kurosawa=is_source, excluded=False,
                              first_indexed_frame=first_name, last_indexed_frame=last_name,
                              indexed_frames=movie["frames"], selected_frames=len(records)))
    del catalog
    write_json(out / "inventory.json", inventory)
    note(f"Inventory: {len(sampled)} included movies, {len(EXCLUDED)} excluded, {sum(map(len, sampled)):,} selected frames")
    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    if not hub.exists() or not checkpoint.exists():
        raise FileNotFoundError("This offline pilot requires the cached DINOv2 repository and small checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("Model checkpoint changed. Update VERSION and CHECKPOINT_SHA256 before reusing caches.")
    model = load_model(hub, checkpoint)
    records, globals_, spatial_ = [], [], []
    rejected = 0
    # Process Kurosawa first, then targets. Each movie is cached for interruption recovery.
    sampled.sort(key=lambda rs: (not rs[0]["kurosawa"], rs[0]["movie"]))
    with ThreadPoolExecutor(4) as pool:
        for n, selected in enumerate(sampled, 1):
            kept, g, p, cached = embed_movie(model, selected, cache, args.batch_size, pool)
            rejected += len(selected) - len(kept)
            records.extend(kept); globals_.append(g); spatial_.append(p)
            if n <= 10 or n % 10 == 0 or n == len(sampled):
                note(f"{n}/{len(sampled)} movies, {len(records):,} frames; {selected[0]['title']}" + (" (cached)" if cached else ""))
    global_v, spatial_v = np.concatenate(globals_), np.concatenate(spatial_)
    np.save(out / "global.npy", global_v.astype(np.float16))
    np.save(out / "spatial.npy", spatial_v.astype(np.float16))
    note("Ranking source shots against distinct earlier/later films")
    queries, shortlist, film_matches = rank(records, global_v, spatial_v, args.topk)
    used = {q["source"] for q in queries}
    for query in queries:
        for mode in ("later", "earlier"):
            used.update(p["target"] for p in query[mode])
    used.update(p["target"] for p in film_matches)
    make_assets(out, records, used)
    summary = dict(version=VERSION, generated=datetime.now(timezone.utc).isoformat(),
                   source_movies=len({r["movie"] for r in records if r["kurosawa"]}),
                   target_movies=len({r["movie"] for r in records if not r["kurosawa"]}),
                   source_frames=sum(r["kurosawa"] for r in records),
                   target_frames=sum(not r["kurosawa"] for r in records),
                   selected_source_shots_per_film=args.source_shots,
                   selected_target_shots_per_film=args.target_shots,
                   blank_frames_rejected=rejected, shortlist_pairs=len(shortlist),
                   later_films_with_matches=len({records[p["target"]]["movie"] for p in film_matches}),
                   index_catalog=str(catalog_path), exclusions=EXCLUDED,
                   model_checkpoint_sha256=checkpoint_sha256,
                   method="Cached DINOv2 ViT-S/14, RGB, aspect-preserving black padding to 224 × 224. Similarity = 0.5 × global-token cosine + 0.5 × mean cosine across four corresponding patch quadrants. Up to five distinct films per reference in each time group. Shortlist adds film/shot diversity; it has not been human validated.",
                   elapsed_seconds=round(time.monotonic() - START, 2))
    result = dict(summary=summary, records=records, queries=queries, shortlist=shortlist,
                  film_matches=film_matches)
    validate(result, global_v, spatial_v)
    write_json(out / "results.json", result)
    write_json(out / "summary.json", summary)
    payload = json.dumps(result).replace("<", "\\u003c")
    (out / "gallery.html").write_text(PAGE.replace("__DATA__", payload))
    contact_sheet(out, records, shortlist)
    note(f"Complete: {out / 'gallery.html'}")


def validate(result, global_v, spatial_v):
    records = result["records"]
    assert len(global_v) == len(spatial_v) == len(records)
    assert np.isfinite(global_v).all() and np.isfinite(spatial_v).all()
    assert np.allclose(np.linalg.norm(global_v.astype(np.float32), axis=1), 1, atol=.002)
    assert np.allclose(np.linalg.norm(spatial_v.astype(np.float32), axis=1), 1, atol=.002)
    for query in result["queries"]:
        source = records[query["source"]]
        assert source["kurosawa"]
        for mode in ("later", "earlier"):
            matches = query[mode]
            assert len({records[p["target"]]["movie"] for p in matches}) == len(matches)
            for pair in matches:
                target = records[pair["target"]]
                assert not target["kurosawa"]
                assert (target["year"] > source["year"] if mode == "later"
                        else target["year"] < source["year"])
                assert abs(pair["score"] - .5 * (pair["global_score"] + pair["spatial_score"])) < .00002
                assert Path(target["path"]).exists()
    for pair in result["shortlist"]:
        assert records[pair["target"]]["year"] > records[pair["source"]]["year"]
    for pair in result["film_matches"]:
        assert records[pair["source"]]["kurosawa"] and not records[pair["target"]]["kurosawa"]
        assert records[pair["target"]]["year"] > records[pair["source"]]["year"]
    assert len({p["source"] for p in result["shortlist"]}) == len(result["shortlist"])
    note("Validated vectors, frame paths, distinct-film retrieval, score arithmetic, and chronology")


if __name__ == "__main__":
    main()
