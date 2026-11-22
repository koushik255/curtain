"""Why the right film is missing from the 20 results, for web_search_v2 misses.

For each clean-frame / overlay miss whose film is not in the returned 20, scan
every frame of the right film (the label as an oracle, diagnosis only):

    pool_rank    where the film's first frame lands in the FAISS candidate pool
                 (before alignment; None = not in the pool at all)
    best frame   the film's best-scoring frame by embedding, its score next to
                 the score of the site's #1, and SIFT agreeing points between
                 the query and that frame (>= 15: the same shot is in our footage)

Then also the best of the film's top 16 frames by SIFT, in case the embedding's
best is the wrong shot but the right one is close behind.

    .venv/bin/python -m benchmarks.web_search_oracle
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local/benchmarks/web-search-v2"
os.environ.setdefault("CURTAIN_FAISS_DIR", str(ROOT / ".local/faiss-all-films-v2"))
os.environ.setdefault("CURTAIN_CHECKPOINT", str(ROOT / "curtain_models/all-films-v1/best.pt"))  # what the site runs


def main():
    import torch
    from apps.server import DEFAULT_CHECKPOINT, TrainedSearchEngine
    from rcmindex import align, search

    torch.set_num_threads(8)
    reviews = json.loads((OUT / "reviews.json").read_text())
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
    cases = [r for r in rows if r["variant"] == "original" and not r["top20"]
             and reviews[r["id"]] in ("frame", "overlay")]
    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT)
    out = []
    for n, r in enumerate(cases, 1):
        with Image.open(OUT / "misses-site/img" / f"{r['id']}.q.jpg") as im:
            image = im.convert("RGB")
        readings = engine.embed_query(image)
        frames, pool_scores = search.candidate_frames(engine.search_index, readings)
        order = np.argsort(-pool_scores, kind="stable")
        film = r["movie"].casefold()
        ranks = [k for k, p in enumerate(order, 1) if engine.records[int(frames[p])]["movie"].casefold() == film]
        top_pool = float(pool_scores[order[0]])
        movie = engine.movie_lookup[r["movie"]]
        indices = np.concatenate([np.arange(s, s + c) for s, c in movie["spans"]])
        scores = np.empty(len(indices), dtype=np.float32)
        for s in range(0, len(indices), 1024):
            batch = indices[s:s + 1024]
            scores[s:s + len(batch)] = search.scores(search.frame_vectors(engine.search_index, batch), readings).max(axis=0)
        features = align.features(np.asarray(image.convert("L")))
        best = []
        for p in np.argsort(-scores, kind="stable")[:16]:
            index = int(indices[p])
            points = engine.locate(features, index)[0]
            best.append({"index": index, "filename": engine.records[index]["filename"],
                         "score": float(scores[p]), "points": int(points)})
        by_points = max(best, key=lambda b: b["points"])
        row = {"id": r["id"], "movie": r["movie"], "kind": reviews[r["id"]], "pool_size": len(frames),
               "pool_rank": ranks[0] if ranks else None, "top_pool_score": top_pool,
               "film_frames": len(indices), "embedding_best": best[0], "sift_best_of_16": by_points}
        out.append(row)
        (OUT / "oracle.json").write_text(json.dumps(out, indent=1))
        print(f"{n}/{len(cases)} {r['id']}: pool rank {row['pool_rank']}, film best {best[0]['score']:.3f} "
              f"vs top {top_pool:.3f}, points {best[0]['points']} / best-of-16 {by_points['points']}", flush=True)


if __name__ == "__main__":
    main()
