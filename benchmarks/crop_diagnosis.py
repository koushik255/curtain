"""Separate crop failures into candidate retrieval, scoring and alignment.

Uses a specified, matching checkpoint/index pair locally. Independent web
queries have movie labels but no asserted exact-frame ground truth. The film
scan is an oracle diagnostic, never included in ordinary search accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def variants(source: Image.Image, center: tuple[float, float]):
    yield "web", source
    for fraction in (0.7, 0.4, 0.25, 0.15):
        w, h = source.size
        cw, ch = max(1, round(w * fraction)), max(1, round(h * fraction))
        x = max(0, min(w - cw, round(w * center[0] - cw / 2)))
        y = max(0, min(h - ch, round(h * center[1] - ch / 2)))
        cut = source.crop((x, y, x + cw, y + ch))
        yield f"crop{round(fraction * 100)}", cut
        if fraction == 0.4:
            buffer = io.BytesIO()
            cut.save(buffer, "JPEG", quality=15)
            yield "crop40-jpeg15", Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
            graded = ImageEnhance.Brightness(ImageEnhance.Color(cut).enhance(0.5)).enhance(0.65)
            yield "crop40-dark", graded
            pixels = np.asarray(cut,dtype=np.float32)/255
            pixels = np.maximum(pixels-0.045,0)/0.955
            pixels = np.clip(pixels*np.array([1.08,0.80,0.88],dtype=np.float32),0,1)**1.4
            release = Image.fromarray((pixels*255).round().astype(np.uint8))
            yield "crop40-release", release
            small = release.resize((160,max(1,round(release.height*160/release.width))),Image.Resampling.BILINEAR)
            buffer=io.BytesIO();small.save(buffer,'JPEG',quality=20)
            yield "crop40-release-lowres",Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    yield "gray", ImageOps.grayscale(source).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/data/godfather_web_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / ".local/benchmarks/godfather-web-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "curtain_models/all-films-v1/best.pt")
    parser.add_argument("--index", type=Path, default=ROOT / ".local/faiss-all-films-v1")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    os.environ["CURTAIN_CHECKPOINT"] = str(args.checkpoint)
    os.environ["CURTAIN_FAISS_DIR"] = str(args.index)
    from apps.server import ALIGN_CANDIDATES, TrainedSearchEngine
    from rcmindex import align, search
    from rcmindex.validation_suite import browser_upload
    import faiss

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    started = time.perf_counter()
    engine = TrainedSearchEngine(args.checkpoint)
    print(f"Loaded {args.checkpoint.parent.name}, {engine.vectors} vectors in {time.perf_counter()-started:.1f}s", flush=True)
    manifest = json.loads(args.manifest.read_text())
    items = [c for c in manifest["queries"] if not args.ids or c["id"] in args.ids]
    film_cache = {}
    report = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "index": str(args.index), "manifest": str(args.manifest),
              "catalog_sha256": hashlib.sha256((args.index / 'catalog.json').read_bytes()).hexdigest(),
              "notes": "Web source frame IDs unlabelled. Film-only scans use expected movie as an oracle. Crop fractions are per side, not area.",
              "rows": []}
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    report_path = args.out / f"diagnosis-{args.checkpoint.parent.name}-{stamp}.json"

    for case in items:
        if case.get('target') and case['target'].split('/')[1] != case['movie']:
            raise ValueError(f"Target movie differs from expected movie for {case['id']}")
        matches = [i for i,m in enumerate(engine.search_index["movies"]) if m["movie"].split("/")[-1] == case["movie"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one indexed copy of {case['movie']}, found {len(matches)}")
        movie_index = matches[0]
        movie = engine.search_index["movies"][movie_index]
        first, count = movie["first"], movie["frames"]
        if movie_index not in film_cache:
            film_cache[movie_index] = np.concatenate([
                engine.search_index["tiles"][movie_index],
                engine.search_index["whole"][movie_index][:,None]],axis=1).astype(np.float32)
        vectors = film_cache[movie_index]
        with Image.open(args.out / "originals" / case["id"]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
        center = tuple(case.get("crop_center", [0.5,0.5]))
        for kind, raw in variants(source, center):
            if args.variants and kind not in args.variants:
                continue
            t = time.perf_counter()
            image = browser_upload(raw)
            queries = args.out / "queries"
            queries.mkdir(exist_ok=True)
            query_path = queries / f"{case['id']}-{kind}.png"
            image.save(query_path)
            readings = engine.embed_query(image)
            frames, scores = search.candidate_frames(engine.search_index, readings)
            order = np.argsort(-scores)
            frames, scores = frames[order], scores[order]
            in_movie = np.flatnonzero((frames >= first) & (frames < first + count))
            movie_scores = search.scores(vectors,readings).max(axis=0)
            film_order = np.argsort(-movie_scores)
            query_features = align.features(np.asarray(image.convert("L")))
            oracle = []
            for local in film_order[:12]:
                frame_id = first + int(local)
                points, matrix, size = engine.locate(query_features,frame_id)
                oracle.append({"index":frame_id, "filename":engine.records[frame_id]["filename"],
                               "score":float(movie_scores[local]), "points":points,
                               "candidate":bool(frame_id in frames),
                               "rank_if_injected":int((scores>movie_scores[local]).sum())+1})
            ranked, _ = engine.rank(image,readings)
            results = [{**engine.records[f],"index":f,"score":score,"points":a[0] if a else 0}
                       for f,score,a in ranked]
            actual = results[0] if results else {}
            row = {"id":case["id"],"movie":case["movie"],"variant":kind,
                   "query":str(query_path.relative_to(args.out)),"source_sha256":hashlib.sha256((args.out/'originals'/case['id']).read_bytes()).hexdigest(),
                   "query_sha256":hashlib.sha256(query_path.read_bytes()).hexdigest(),
                   "query_size":image.size,"candidate_count":len(frames),
                   "movie_candidate_rank":int(in_movie[0])+1 if len(in_movie) else None,
                   "movie_candidate_score":float(scores[in_movie[0]]) if len(in_movie) else None,
                   "alignment_cutoff":ALIGN_CANDIDATES,"candidate_cutoff_score":float(scores[min(ALIGN_CANDIDATES,len(scores))-1]),
                   "film_scan":oracle,"top":actual,"results":results,
                   "right_movie":actual.get("movie")==case["movie"],
                   "seconds":time.perf_counter()-t}
            if target := case.get("target"):
                target_id = engine.frame_index(target)
                if target_id is None or not first <= target_id < first+count:
                    raise ValueError(f"Unindexed target {target}")
                target_score = float(movie_scores[target_id-first])
                row["target"] = target
                row["target_score"] = target_score
                row["target_candidate"] = bool(target_id in frames)
                row["target_rank_if_injected"] = int((scores > target_score).sum())+1
                row["target_film_rank"] = int((movie_scores > target_score).sum())+1
                row["exact_top1"] = actual.get("index") == target_id
                row["exact_top5"] = any(r["index"] == target_id for r in results[:5])
            report["rows"].append(row)
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            print(f"{case['id']} {kind}: {'RIGHT' if row['right_movie'] else 'WRONG'}; "
                  f"movie candidate rank {row['movie_candidate_rank']}; oracle {oracle[0]['filename']} "
                  f"{oracle[0]['score']:.3f} ({oracle[0]['points']} SIFT), would rank {oracle[0]['rank_if_injected']}; "
                  f"top {actual.get('movie')} {actual.get('filename')} ({actual.get('points')} SIFT)",flush=True)
    make_sheet(report, args.out, engine, report_path.with_suffix('.jpg'))
    print(f"Saved {report_path}",flush=True)


def make_sheet(report, out, engine, path):
    rows = report['rows']
    sheet=Image.new('RGB',(1200,210*len(rows)),'#222222');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        top=i*210
        images=[('query',out/row['query']),
                ('ordinary search',engine.frame_path(row['top']['index'])),
                ('known-film oracle',engine.frame_path(row['film_scan'][0]['index']))]
        for col,(label,p) in enumerate(images):
            if p is None:continue
            with Image.open(p) as im:image=ImageOps.contain(im.convert('RGB'),(390,170))
            sheet.paste(image,(400*col,top+22));draw.text((400*col,top+5),label,fill='white')
        draw.text((0,top+192),f"{row['id']} {row['variant']} | right movie: {row['right_movie']} | "
                  f"movie candidate rank: {row['movie_candidate_rank']}",fill='white')
    sheet.save(path,quality=90)


if __name__=='__main__':
    main()
