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
import operator
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

# The index the site and benchmarks search; CURTAIN_FAISS_DIR picks another
# (build_faiss.py --out), e.g. one built from a new model's arrays.
FAISS_DIR = Path(os.environ.get("CURTAIN_FAISS_DIR", Path(__file__).resolve().parents[1] / ".local/faiss"))
# On-disk copy of tiles4-global-ivf16384-pq32.faiss (rcmindex/ondisk_faiss.py):
# its inverted lists (~14 GB of PQ codes and ids) live in the .ivfdata file
# beside it and are memory-mapped, so they sit in the page cache, not the heap.
# Opened with IO_FLAG_READ_ONLY (the .ivfdata is mapped read-only either way);
# IO_FLAG_MMAP on this format segfaults on search with faiss 1.15.1.
INDEX_NAME = "tiles4-global-ivf16384-pq32.ondisk.faiss"
# On the 400-query benchmark, mean-only scoring found 52.5% of full frames and
# whole-only 13.2% of crops; max(whole - 0.1, mean) keeps 93.5% and 48.0%.
WHOLE_HANDICAP = 0.1
READ_THREADS = 8
# FAISS tiles per query image; their frames are re-scored exactly. 128 tiles
# often cover only a few frames (a frame has 50 vectors), so a shot whose
# frame differs in a small region (a turned head) can miss the cut. 512 reads
# ~4x the frames (~2,200 vs ~600): FAISS + re-scoring ~110 ms vs ~70 ms on
# this machine. rank_query.py shows where the right frame lands.
# Override with CURTAIN_CANDIDATES.
CANDIDATES = int(os.environ.get("CURTAIN_CANDIDATES", 512))
# FAISS's approximate (PQ) scores can drop the right frame while keeping one
# a few frames away in the same shot (2 fps), so the NEIGHBOUR_TOP best
# re-scored frames also bring in their frames within NEIGHBOURS either side.
# A Godfather shot whose frame differs only in the face: right frame at #2
# instead of not fetched; ~100 more frames to read, ~5 ms.
NEIGHBOUR_TOP = 50
NEIGHBOURS = 4
READERS = ThreadPoolExecutor(READ_THREADS)


def pieces(image: Image.Image) -> list[Image.Image]:
    """Four overlapping corner pieces of a query, each 60% of its width and height."""
    w, h = image.size
    pw, ph = int(w * 0.6), int(h * 0.6)
    return [image.crop((x, y, x + pw, y + ph)) for y in (0, h - ph) for x in (0, w - pw)]


# Letterbox bars: whole rows/columns at an edge that are uniformly black. A dark
# scene still has a highlight or noise somewhere along a full row; JPEG ringing
# next to the picture can lift a bar's last row, which then just stays.
BAR_MAX, BAR_STD = 28, 4.0
BAR_MIN = 0.02  # thinner than this (of the side) is a dark edge, not a bar


def trim_bars(image: Image.Image) -> Image.Image:
    """The query without black letterbox/pillarbox bars (unchanged if none, or
    if they would leave less than a quarter of a side, e.g. a black frame)."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    def bars(lines: np.ndarray) -> int:
        count = 0
        for line in lines:
            if line.max() > BAR_MAX or line.std() > BAR_STD:
                break
            count += 1
        return count if count >= BAR_MIN * len(lines) else 0
    h, w = gray.shape
    top, bottom, left, right = bars(gray), bars(gray[::-1]), bars(gray.T), bars(gray.T[::-1])
    if h - top - bottom < h / 4:
        top = bottom = 0
    if w - left - right < w / 4:
        left = right = 0
    return image.crop((left, top, w - right, h - bottom)) if top or bottom or left or right else image


# Index tiles cover frame-shaped regions (a quarter of each side) squashed to
# 192x192, so a sliver much taller or wider than a frame is searched as a strip
# of overlapping frame-shaped pieces rather than squashed whole.
FRAME_ASPECT = 16 / 9
MAX_STRIPS = 8


def strips(image: Image.Image) -> list[Image.Image] | None:
    """Frame-shaped pieces along a sliver at half-piece steps, or None if the
    image is roughly frame-shaped (then the corner pieces apply)."""
    w, h = image.size
    tall = w / h < FRAME_ASPECT / 2
    if not tall and w / h <= FRAME_ASPECT * 2:
        return None
    side, length = (w, h) if tall else (h, w)
    piece = side / FRAME_ASPECT if tall else side * FRAME_ASPECT
    count = min(MAX_STRIPS, max(2, int(np.ceil((length - piece) / (piece / 2))) + 1))
    starts = np.linspace(0, length - piece, count).round().astype(int)
    size = int(round(piece))
    return [image.crop((0, s, w, s + size)) if tall else image.crop((s, 0, s + size, h)) for s in starts]


def levels(image: Image.Image) -> Image.Image:
    """Each channel stretched so its 1st-99th percentile spans 0-255: auto
    levels, which also takes out a colour cast."""
    pixels = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(pixels.reshape(-1, 3), [1, 99], axis=0)
    out = (pixels - low) / np.maximum(high - low, 1) * 255
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def grid(image: Image.Image, n: int = 3, size: float = 0.45) -> list[Image.Image]:
    """An n x n grid of overlapping pieces, each `size` of the width and height."""
    w, h = image.size
    pw, ph = int(w * size), int(h * size)
    return [image.crop((x, y, x + pw, y + ph))
            for y in np.linspace(0, h - ph, n).astype(int) for x in np.linspace(0, w - pw, n).astype(int)]


# The levels reading (benchmark_ideas.py, 470 queries on 63 films): exact
# frame 348 -> 356 with the original model (colour-shifted 13 -> 19 of 40),
# but 401 -> 399 with colour-ft-v1, which already ignores grading. Turn it off
# (CURTAIN_COLOUR_READING=0) with a colour-robust model's index.
COLOUR_READING = os.environ.get("CURTAIN_COLOUR_READING", "1") != "0"


def query_sets(image: Image.Image, colour: bool = COLOUR_READING, finer: bool = True) -> list[list[Image.Image]]:
    """Readings of one query, each a list of images to embed with the whole
    first: as uploaded with its corner pieces; then, only where they apply,
    with letterbox bars trimmed, and as a sliver's frame-shaped strips; then
    with auto levels (colour) and with a 3x3 grid of smaller pieces (finer;
    exact frame 348 -> 353 and 401 -> 406, mostly wide crops and subtitles).
    search() keeps each frame's best score over the readings, so a reading
    that does not apply (a dark scene's black edge trimmed, bars that are part
    of the indexed frame) cannot push the right frame down."""
    sets = [[image, *pieces(image)]]
    trimmed = trim_bars(image)
    if trimmed is not image:
        sets.append([trimmed, *pieces(trimmed)])
    for reading in [image] if trimmed is image else [image, trimmed]:
        if parts := strips(reading):
            sets.append([reading, *parts])
    if colour:
        balanced = levels(image)
        sets.append([balanced, *pieces(balanced)])
    if finer:
        sets.append([image, *grid(image)])
    return sets


class FrameNames(Sequence):
    """catalog.json "names" ("<collection>/<movie>/<frame>.jpg" in frame-id
    order), stored as one fixed-width bytes array of frame filenames plus the
    per-movie spans; 7M separate Python strings took about 1 GB."""

    def __init__(self, movies: list[dict], names: list[str]):
        self.movies = [m["movie"] for m in movies]
        self.firsts = np.array([m["first"] for m in movies], dtype=np.int64)
        files = []
        for m in movies:
            if m["first"] != len(files):
                raise RuntimeError("Catalog frame ids are not contiguous.")
            prefix = m["movie"] + "/"
            for name in names[m["first"]:m["first"] + m["frames"]]:
                if not name.startswith(prefix) or "/" in name[len(prefix):]:
                    raise RuntimeError(f"Invalid frame name {name}.")
                files.append(name[len(prefix):].encode())
        if len(files) != len(names):
            raise RuntimeError("Catalog names and movies disagree.")
        self.files = np.array(files, dtype=np.bytes_)  # sorted within each movie

    def __len__(self) -> int:
        return len(self.files)

    def row(self, index: int) -> int:
        row = operator.index(index)
        if row < 0:
            row += len(self.files)
        if not 0 <= row < len(self.files):
            raise IndexError("frame id out of range")
        return row

    def movie_of(self, index: int) -> int:
        """Position in catalog["movies"] of the movie holding this frame."""
        return int(np.searchsorted(self.firsts, self.row(index), side="right")) - 1

    def filename(self, index: int) -> str:
        return self.files[self.row(index)].decode()

    def find(self, start: int, count: int, filename: str) -> int | None:
        """Frame id of filename within the span [start, start + count), if present."""
        try:
            key = filename.encode()
        except UnicodeEncodeError:
            return None
        if not key or len(key) > self.files.itemsize or key.endswith(b"\0"):
            return None
        span = self.files[start:start + count]
        row = int(np.searchsorted(span, key))
        return start + row if row < len(span) and span[row] == key else None

    def __getitem__(self, index: int) -> str:
        return f"{self.movies[self.movie_of(index)]}/{self.filename(index)}"


def load(directory: Path = FAISS_DIR, nprobe: int = 64) -> dict:
    catalog = json.loads((directory / "catalog.json").read_text())
    index = faiss.read_index(str(directory / INDEX_NAME), faiss.IO_FLAG_READ_ONLY)
    faiss.extract_index_ivf(index).nprobe = nprobe
    movies = catalog["movies"]
    names = FrameNames(movies, catalog.pop("names"))
    return {
        "index": index,
        "names": names,
        "movies": movies,
        "stride": catalog["id_stride"],
        "firsts": np.array([m["first"] for m in movies]),
        "ends": np.array([m["first"] + m["frames"] for m in movies]),
        "tiles": [np.load(m["tiles"], mmap_mode="r") for m in movies],
        "whole": [np.load(m["global"], mmap_mode="r") for m in movies],
    }


def frame_vectors(engine: dict, frames: np.ndarray) -> np.ndarray:
    """[len(frames), 50, 128] float32: 49 tiles then the whole frame.

    The arrays (~90 GB) don't fit in RAM, so most rows are page faults on the
    NVMe; READERS threads fault in parallel (4x faster than one at a time)."""
    movie = np.searchsorted(engine["firsts"], frames, side="right") - 1
    rows = frames - engine["firsts"][movie]
    out = np.empty((len(frames), 50, 128), dtype=np.float32)

    def read(chunk: np.ndarray) -> None:
        for m in np.unique(movie[chunk]):
            pick = chunk[movie[chunk] == m]  # rows ascending: sorted reads from the mmap
            out[pick, :49] = engine["tiles"][m][rows[pick]]
            out[pick, 49] = engine["whole"][m][rows[pick]]
    list(READERS.map(read, np.array_split(np.lexsort((rows, movie)), READ_THREADS)))
    return out


def scores(vectors: np.ndarray, queries: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Exact score of each frame's [50, 128] vectors for queries (the whole
    image first), or [readings, frames] for a list of readings. One matrix
    product for all of them (a batched 3-D one per reading is ~4x slower)."""
    readings = queries if isinstance(queries, list) else [queries]
    views = np.concatenate(readings)
    best = (vectors.reshape(-1, 128) @ views.T).reshape(len(vectors), 50, len(views)).max(axis=1)  # [f, q]
    bounds = np.cumsum([0, *map(len, readings)])
    out = np.stack([np.maximum(best[:, a] - WHOLE_HANDICAP, best[:, a:b].mean(axis=1))
                    for a, b in zip(bounds, bounds[1:])])
    return out if isinstance(queries, list) else out[0]


def rescore(engine: dict, queries: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Exact score of each frame id for queries (the whole image first)."""
    return scores(frame_vectors(engine, frames), queries)


def neighbours(engine: dict, frames: np.ndarray, reach: int = NEIGHBOURS) -> np.ndarray:
    """Frame ids within reach of each frame, in the same movie."""
    movie = np.searchsorted(engine["firsts"], frames, side="right") - 1
    near = [np.clip(frames + d, engine["firsts"][movie], engine["ends"][movie] - 1)
            for d in range(-reach, reach + 1)]
    return np.unique(np.concatenate(near))


def candidate_frames(engine: dict, queries: np.ndarray | list[np.ndarray],
                     count: int = CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """(frame ids, exact scores): the frames of the count FAISS tiles per
    query image, plus the neighbours of the best NEIGHBOUR_TOP of them. Given
    several readings of a query (query_sets), every candidate is scored for
    every reading and keeps its best score; the neighbours are those of each
    reading's best. The readings share most candidates, so each frame's
    vectors are read once (~2x faster than a search per reading, 4 readings)."""
    readings = queries if isinstance(queries, list) else [queries]
    readings = [np.ascontiguousarray(q, dtype=np.float32) for q in readings]
    _, ids = engine["index"].search(np.concatenate(readings), count)  # one batch for all views
    frames = np.unique(ids[ids >= 0] // engine["stride"])
    if len(frames) == 0:
        return frames, np.empty(0, dtype=np.float32)

    each = scores(frame_vectors(engine, frames), readings)  # [readings, frames]
    best = np.unique(frames[np.argsort(-each, axis=1)[:, :NEIGHBOUR_TOP]])
    extra = np.setdiff1d(neighbours(engine, best), frames)
    return (np.concatenate([frames, extra]),
            np.concatenate([each.max(axis=0), scores(frame_vectors(engine, extra), readings).max(axis=0)]))


def search(engine: dict, queries: np.ndarray | list[np.ndarray], candidates: int = CANDIDATES,
           top: int = 5) -> list[tuple[int, float]]:
    """queries: [q, 128] unit vectors for one search, the whole query image
    first, then its pieces; or a list of those, one per reading (query_sets).
    Returns [(frame id, score)] best first."""
    if not isinstance(queries, list):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
    frames, score = candidate_frames(engine, queries, candidates)
    order = np.argsort(-score)[:top]
    return [(int(frames[i]), float(score[i])) for i in order]
