"""Where does a query's movie (or frame) land in the search, before alignment?

Loads the site's engine (apps.server.TrainedSearchEngine) and runs one query
as the site does: the image as the page uploads it (640 px, JPEG 90), the whole
image plus 4 pieces, `--candidates` FAISS tiles per view, exact re-scoring
(with the neighbours of the best frames, search.candidate_frames).
Prints, for each view, whether the expected movie is among its FAISS tiles;
the rank and score of the movie's best frame after re-scoring; the top frames;
and with --frame, that frame's own exact score and the rank it would get.
--scan re-scores every frame of the movie, to see if the shot can score high
at all (FAISS recall) or not (the embedding). --align shows the site's order
after SIFT alignment.

    .venv/bin/python -m rcmindex.rank_query query.png The_Godfather_1972
    .venv/bin/python -m rcmindex.rank_query query.png The_Godfather_1972 \\
        --frame frame_004812.jpg --candidates 128 --scan --align
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from apps.server import ALIGN_CANDIDATES, DEFAULT_CHECKPOINT, TrainedSearchEngine
from curtain_ml.retrieval import normalize_rows
from rcmindex import align, search
from rcmindex.validation_suite import browser_upload

SCAN_CHUNK = 4096  # frames re-scored at a time with --scan (~100 MB of vectors)


def find_movie(engine: TrainedSearchEngine, name: str) -> dict:
    # Catalog entry by "<collection>/<movie>" or just "<movie>" (any case).
    movies = engine.search_index["movies"]
    exact = [m for m in movies if name.lower() in (m["movie"].lower(), m["movie"].split("/")[1].lower())]
    if len(exact) == 1:
        return exact[0]
    close = [m["movie"] for m in movies if name.lower() in m["movie"].lower()]
    raise SystemExit(f"movie {name!r} {'is ambiguous' if exact else 'not found'}; "
                     f"close: {', '.join(close[:10]) or 'none'}")


def label(engine: TrainedSearchEngine, frame: int) -> str:
    return engine.search_index["names"][frame]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("movie", help="expected movie folder, e.g. The_Godfather_1972")
    parser.add_argument("--frame", help="expected frame filename, e.g. frame_004812.jpg")
    parser.add_argument("--candidates", type=int, default=search.CANDIDATES, help="FAISS tiles per view")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--raw", action="store_true", help="skip the page's 640 px JPEG upload step")
    parser.add_argument("--scan", action="store_true", help="also re-score every frame of the movie")
    parser.add_argument("--align", action="store_true", help=f"also show the order after aligning up to {ALIGN_CANDIDATES}")
    args = parser.parse_args()

    started = time.perf_counter()
    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT)
    index = engine.search_index
    print(f"engine loaded in {time.perf_counter() - started:.0f}s")
    movie = find_movie(engine, args.movie)
    first, count = movie["first"], movie["frames"]
    target = None
    if args.frame:
        target = index["names"].find(first, count, args.frame)
        if target is None:
            raise SystemExit(f"{args.frame} is not an indexed frame of {movie['movie']}")

    with Image.open(args.image) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if not args.raw:
        image = browser_upload(image).convert("RGB")
    views = [image, *search.pieces(image)]
    tensor = torch.stack([engine.transform(view) for view in views]).to(engine.device)
    with torch.inference_mode():
        queries = np.ascontiguousarray(normalize_rows(engine.model(tensor).float().cpu().numpy()), dtype=np.float32)

    print(f"\nquery {image.size[0]}x{image.size[1]}, expected {movie['movie']} "
          f"(frames {first}-{first + count - 1})" + (f", frame {target}" if target is not None else ""))
    scores, ids = index["index"].search(queries, args.candidates)
    frame_ids = ids // index["stride"]
    print(f"\nFAISS: {args.candidates} tiles per view")
    for v, name in enumerate(["whole", "piece TL", "piece TR", "piece BL", "piece BR"]):
        valid = ids[v] >= 0
        hits = np.flatnonzero(valid & (frame_ids[v] >= first) & (frame_ids[v] < first + count))
        line = f"  {name:9s} {len(np.unique(frame_ids[v][valid])):4d} frames;"
        if len(hits):
            line += (f" movie: {len(hits)} tiles, {len(np.unique(frame_ids[v][hits]))} frames, "
                     f"best at #{hits[0] + 1} ({scores[v][hits[0]]:.3f}; #1 is {scores[v][0]:.3f})")
            if target is not None:
                near = np.abs(frame_ids[v][hits] - target).min()
                line += f", {'has the frame' if near == 0 else f'nearest {near} frames away'}"
        else:
            line += f" movie: none (cut-off {scores[v][valid][-1]:.3f})" if valid.any() else " no results"
        print(line)

    frames, score = search.candidate_frames(index, queries, args.candidates)
    fetched = len(np.unique(ids[ids >= 0] // index["stride"]))
    order = np.argsort(-score)
    ranked, ranked_score = frames[order], score[order]
    in_movie = np.flatnonzero((ranked >= first) & (ranked < first + count))
    print(f"\nre-scored {len(frames)} candidate frames ({fetched} from FAISS, {len(frames) - fetched} neighbours)")
    if len(in_movie):
        r = in_movie[0]
        print(f"  movie's best frame: rank {r + 1} of {len(frames)}, {label(engine, ranked[r])} ({ranked_score[r]:.4f})")
    else:
        print("  movie's best frame: not among the candidates")
    if target is not None:
        own = float(search.rescore(index, queries, np.array([target]))[0])
        rank = int((ranked_score > own).sum()) + 1
        where = "a candidate" if target in frames else "NOT a candidate"
        print(f"  expected frame: {own:.4f}, would rank {rank} ({where})")
    print(f"\ntop {args.top}:")
    for r in range(min(args.top, len(ranked))):
        mark = "*" if first <= ranked[r] < first + count else " "
        print(f" {mark}{r + 1:3d}  {ranked_score[r]:.4f}  {label(engine, ranked[r])}")

    if args.scan:
        every = np.arange(first, first + count)
        movie_score = np.concatenate([search.rescore(index, queries, every[i:i + SCAN_CHUNK])
                                      for i in range(0, count, SCAN_CHUNK)])
        best = np.argsort(-movie_score)[:5]
        print(f"\nscan of all {count} frames of the movie (rank among the candidates if they had been one):")
        for b in best:
            frame = int(every[b])
            note = "candidate" if frame in frames else "not a candidate"
            print(f"  {movie_score[b]:.4f}  rank {int((ranked_score > movie_score[b]).sum()) + 1:4d}  "
                  f"{label(engine, frame)} ({note})")

    if args.align:
        found = [(int(f), float(s)) for f, s in zip(ranked[:ALIGN_CANDIDATES], ranked_score[:ALIGN_CANDIDATES])]
        t = time.perf_counter()
        alignments = engine.align_candidates(image, found)
        elapsed = (time.perf_counter() - t) * 1000
        print(f"\nalignment: {len(alignments)} of {len(found)} checked in {elapsed:.0f} ms")
        for i in align.rerank(found[:len(alignments)], alignments)[:args.top]:
            mark = "*" if first <= found[i][0] < first + count else " "
            ok = "confirmed" if alignments[i][0] >= align.VERIFIED_POINTS else ""
            print(f" {mark} was {i + 1:3d}  {alignments[i][0]:4d} points {ok:9s} {label(engine, found[i][0])}")


if __name__ == "__main__":
    main()
