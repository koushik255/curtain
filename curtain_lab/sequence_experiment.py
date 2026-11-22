"""Offline sequence retrieval baseline: v1 composition + ordered coarse optical flow.

Four consecutive 2fps frames, wholly within detected shot boundaries. Optical
flow at this sampling rate is approximate apparent image motion, not calibrated
camera motion. No new temporal model is trained. Existing caches stay untouched.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F

import composition_model as cm
import kurosawa_experiment as pilot
from focused_experiment import TARGETS

VERSION = 'sequence-flow-grid4-4frames-v1'


def clip_records(root, catalog, records, per_film):
    first = {m['movie']: m['first'] for m in catalog['movies']}
    groups = {}
    for i, r in enumerate(records):
        if r['shot_end_row'] - r['shot_start_row'] >= 4:
            groups.setdefault(r['movie'], []).append((i, r))
    clips = []
    for movie, candidates in groups.items():
        ids = np.unique(np.linspace(0, len(candidates)-1, min(per_film, len(candidates))).round().astype(int))
        for j in ids:
            i, r = candidates[j]
            start = max(r['shot_start_row'], min(r['row'] - 1, r['shot_end_row'] - 4))
            names = catalog['names'][first[movie] + start:first[movie] + start + 4]
            numbers = [int(Path(n).stem.split('_')[-1]) for n in names]
            if np.diff(numbers).tolist() != [1, 1, 1]:
                raise ValueError(f'Nonconsecutive extraction timestamps: {names}')
            clips.append(dict(r, record_index=i, frames=names,
                              paths=[str(pilot.frame_path(root, n)) for n in names],
                              clip_start=pilot.clock(numbers[0]/2), clip_seconds=[n/2 for n in numbers]))
    return clips


def motion(clip):
    # Preserve aspect ratio in a fixed grid, without including black padding.
    images = []
    for path in clip['paths']:
        with Image.open(path) as im:
            images.append(cv2.resize(np.asarray(im.convert('L')), (192, 108)))
    fields = []
    for a, b in zip(images, images[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 3, 5, 1.2, 0)
        # Velocities in fraction of image width/height per second.
        flow *= np.array([2/192, 2/108], np.float32)
        cells = flow.reshape(4, 27, 4, 48, 2).mean(axis=(1, 3))
        fields.append(cells)
    fields = np.asarray(fields)
    # Keep direction, position and chronological interval. Soft compression
    # limits domination by a single large/discontinuous flow estimate.
    vector = np.arcsinh(fields / .01).flatten()
    return vector.astype(np.float32), float(np.linalg.norm(fields, axis=-1).mean())


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def gif_pair(out, name, source, target):
    frames = []
    for k in range(4):
        canvas = Image.new('RGB', (800, 270), '#171b22')
        d = ImageDraw.Draw(canvas)
        for col, clip in enumerate((source, target)):
            with Image.open(clip['paths'][k]) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (390, 220))
            canvas.paste(thumb, (col*400+(400-thumb.width)//2, 0))
            d.text((col*400+8, 225), clip['title'][:40], fill='white')
            d.text((col*400+8, 242), f"{clip['clip_start']}  +{k*.5:.1f}s", fill='white')
        frames.append(canvas)
    frames[0].save(out/name, save_all=True, append_images=frames[1:], duration=500, loop=0)


def run(args):
    torch.set_num_threads(args.threads); cv2.setNumThreads(1)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    records = json.loads((cm.OUT/'search-records.json').read_text())
    catalog = json.loads((args.root/'.local/faiss-all-films-v1/catalog.json').read_text())
    clips = clip_records(args.root, catalog, records, args.per_film)
    signature = hashlib.sha256(json.dumps([VERSION, [(c['frames'], [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in c['paths']]) for c in clips]]).encode()).hexdigest()
    cache = out/f'motion-{signature}.npz'
    if cache.exists():
        with np.load(cache) as data:
            vectors, energy = data['vectors'], data['energy']
    else:
        values = []
        with ThreadPoolExecutor(args.threads) as pool:
            for i, value in enumerate(pool.map(motion, clips), 1):
                values.append(value)
                if i % 100 == 0: pilot.note(f'motion {i}/{len(clips)}')
        vectors = np.stack([v[0] for v in values]); energy = np.array([v[1] for v in values])
        np.savez(cache, vectors=vectors, energy=energy)
    head = cm.CompositionHead()
    head.load_state_dict(torch.load(cm.OUT/'head.pt', map_location='cpu', weights_only=True)); head.eval()
    grid = np.load(cm.OUT/'search-grid.npy', mmap_mode='r')
    with torch.inference_mode():
        z = torch.cat([head(torch.from_numpy(np.array(grid[[c['record_index'] for c in clips[s:s+256]], 0], dtype=np.float32))) for s in range(0,len(clips),256)]).numpy()
    mv = unit(vectors)
    source = np.array([i for i,c in enumerate(clips) if c['kurosawa']])
    rows = []; results = {}; comparison = {}
    for stem in TARGETS:
        target = np.array([i for i,c in enumerate(clips) if c['movie'].endswith('/'+stem)])
        appearance = z[source] @ z[target].T
        movement = mv[source] @ mv[target].T
        # Static clips contribute no motion evidence; matching two static images
        # must not count as a movement match. Threshold is a heuristic, logged.
        active = (energy[source,None] >= .002) & (energy[target][None,:] >= .002)
        movement = np.where(active, movement, 0)
        score = .65*appearance + .35*movement
        def pick(scores):
            found=[]; used_s=[]; used_t=[]; counts={}
            for k in np.argsort(scores, axis=None)[::-1]:
                a,b=divmod(int(k),len(target)); s,t=int(source[a]),int(target[b])
                sr,tr=clips[s],clips[t]
                if counts.get(sr['movie'],0)>=2: continue
                if any(clips[x]['movie']==sr['movie'] and abs(clips[x]['shot']-sr['shot'])<3 for x in used_s): continue
                if any(abs(clips[x]['shot']-tr['shot'])<3 for x in used_t): continue
                found.append((s,t,a,b)); used_s.append(s); used_t.append(t)
                counts[sr['movie']]=counts.get(sr['movie'],0)+1
                if len(found)==6: break
            return found
        chosen=pick(score); still=pick(appearance)
        comparison[stem]={'same_top6_pairs':len({(s,t) for s,t,_,_ in chosen}&{(s,t) for s,t,_,_ in still}),
                          'composition_only_pairs':[(s,t) for s,t,_,_ in still]}
        results[stem]=[]
        for rank,(s,t,a,b) in enumerate(chosen,1):
            name=f'{stem}-{rank}.gif'; gif_pair(out,name,clips[s],clips[t])
            entry=dict(source=s,target=t,score=float(score[a,b]),composition=float(appearance[a,b]),motion=float(movement[a,b]),motion_active=bool(active[a,b]),gif=name)
            results[stem].append(entry)
            rows.append(f'<article><h3>{html.escape(stem)} #{rank}</h3><img loading="lazy" src="{name}"><p>Combined {entry["score"]:.3f} · composition {entry["composition"]:.3f} · motion {entry["motion"]:.3f} · motion active: {entry["motion_active"]}</p></article>')
    pilot.write_json(out/'clips.json',clips)
    pilot.write_json(out/'search.json',results)
    report=dict(version=VERSION, clips=len(clips), films=len({c['movie'] for c in clips}), frames_per_clip=4, fps=2, duration_seconds=1.5,
                active_motion_clips=int((energy>=.002).sum()), motion_threshold=.002,
                composition_weight=.65,motion_weight=.35,comparison=comparison,
                limitations=['Sampled shots, not exhaustive coverage','No temporal training; v1 midpoint composition plus ordered optical flow', 'Optical flow at 2fps can alias fast movement and cannot distinguish actor from camera motion','Detected shot boundaries and extraction clocks are not edition-verified','No human validation or influence claims'])
    pilot.write_json(out/'report.json',report)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sequence experiment</title><style>body{background:#171b22;color:#eee;font:16px sans-serif;max-width:850px;margin:30px auto}img{width:100%}article{margin:30px 0}</style><h1>Composition + movement: sequence baseline</h1><p>Four consecutive screenshots, 0.5 seconds apart, within a detected shot. GIF loops reset after 1.5 seconds. Scores are retrieval heuristics, not evidence of influence or matching camera technique.</p>'+''.join(rows))
    pilot.note(f'Done: {out}; {len(clips)} clips, {len(rows)} animated pairs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=cm.HERE.parent)
    p.add_argument('--out',type=Path,default=cm.HERE/'results/sequence-v1')
    p.add_argument('--per-film',type=int,default=80)
    p.add_argument('--threads',type=int,default=8)
    run(p.parse_args())
