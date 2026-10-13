"""Tile search: FAISS candidates, then exact re-scoring from the fp16 vectors.

A query is one image embedding or several (the crop plus its pieces). Each
is searched in the IVF-PQ index; the union of candidate frames is re-scored
exactly from the memory-mapped rcmindex arrays (best of the frame's 49 tiles
and whole-frame vector, per query image). A frame's score is the better of
the mean over all query images (crops match piece by piece) and the whole
query's score minus WHOLE_HANDICAP (uncropped screenshots match as a whole).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

FAISS_DIR = Path(__file__).resolve().parents[1] / ".local/faiss"
INDEX_NAME = "tiles4-global-ivf16384-pq32.faiss"
# On the 400-query benchmark, mean-only scoring found 52.5% of full frames and
# whole-only 13.2% of crops; max(whole - 0.1, mean) keeps 93.5% and 48.0%.
WHOLE_HANDICAP = 0.1


def pieces(image: Image.Image) -> list[Image.Image]:
    """Four overlapping corner pieces of a query, each 60% of its width and height."""
    w, h = image.size
    pw, ph = int(w * 0.6), int(h * 0.6)
    return [image.crop((x, y, x + pw, y + ph)) for y in (0, h - ph) for x in (0, w - pw)]


def load(directory: Path = FAISS_DIR, nprobe: int = 64) -> dict:
    catalog = json.loads((directory / "catalog.json").read_text())
    index = faiss.read_index(str(directory / INDEX_NAME))
    faiss.extract_index_ivf(index).nprobe = nprobe
    movies = catalog["movies"]
    return {
        "index": index,
        "names": catalog["names"],
        "stride": catalog["id_stride"],
        "firsts": np.array([m["first"] for m in movies]),
        "tiles": [np.load(m["tiles"], mmap_mode="r") for m in movies],
        "whole": [np.load(m["global"], mmap_mode="r") for m in movies],
    }


def frame_vectors(engine: dict, frames: np.ndarray) -> np.ndarray:
    """[len(frames), 50, 128] float32: 49 tiles then the whole frame."""
    movie = np.searchsorted(engine["firsts"], frames, side="right") - 1
    rows = frames - engine["firsts"][movie]
    out = np.empty((len(frames), 50, 128), dtype=np.float32)
    for m in np.unique(movie):
        pick = np.where(movie == m)[0]
        order = np.argsort(rows[pick])  # sorted reads from the mmap
        r = rows[pick][order]
        out[pick[order], :49] = engine["tiles"][m][r]
        out[pick[order], 49] = engine["whole"][m][r]
    return out


def search(engine: dict, queries: np.ndarray, candidates: int = 256, top: int = 5) -> list[tuple[int, float]]:
    """queries: [q, 128] unit vectors for one search, the whole query image
    first, then its pieces. Returns [(frame id, score)] best first."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    _, ids = engine["index"].search(queries, candidates)
    frames = np.unique(ids[ids >= 0] // engine["stride"])
    if len(frames) == 0:
        return []
    vectors = frame_vectors(engine, frames)                  # [f, 50, 128]
    best = (vectors @ queries.T).max(axis=1)                 # [f, q] best view per query image
    score = np.maximum(best[:, 0] - WHOLE_HANDICAP, best.mean(axis=1))
    order = np.argsort(-score)[:top]
    return [(int(frames[i]), float(score[i])) for i in order]
