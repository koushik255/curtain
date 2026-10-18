"""Copy the FAISS tile index into an on-disk (memory-mapped) copy.

The IVF-PQ index from build_faiss.py keeps all 351M PQ codes and ids in RAM
(~14 GB). This writes the same index with its inverted lists moved to an
.ivfdata file (OnDiskInvertedLists), so search.load() can memory-map it and
the kernel pages lists in from NVMe on demand. Nothing is retrained: the
quantizer, PQ codebooks, codes and ids are copied as they are. The source
index is opened read-only (IO_FLAG_MMAP) and left untouched.

Open the result with IO_FLAG_READ_ONLY: an OnDiskInvertedLists index always
memory-maps its .ivfdata, and with faiss 1.15.1 adding IO_FLAG_MMAP to that
read makes searches segfault.

    nice -n 19 .venv/bin/python -m rcmindex.ondisk_faiss
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np

FAISS_DIR = Path(__file__).resolve().parents[1] / ".local/faiss"
SOURCE = FAISS_DIR / "tiles4-global-ivf16384-pq32.faiss"
started = time.monotonic()


def note(message: str) -> None:
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def ondisk_paths(source: Path) -> tuple[Path, Path]:
    """<name>.ondisk.faiss and <name>.ivfdata next to the source index."""
    return source.with_suffix(".ondisk.faiss"), source.with_suffix(".ivfdata")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-queries", type=int, default=200)
    parser.add_argument("--check-only", action="store_true", help="only compare an existing copy")
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)
    index_path, data_path = ondisk_paths(args.source)
    if args.check_only:
        check(args.source, index_path, args.check_queries)
        return
    if index_path.exists() or data_path.exists():
        raise SystemExit(f"{index_path} or {data_path} already exists; remove them first.")

    # IO_FLAG_MMAP maps the source lists read-only instead of loading them.
    source = faiss.read_index(str(args.source), faiss.IO_FLAG_MMAP)
    ivf = faiss.extract_index_ivf(source)
    note(f"source: {source.ntotal:,} vectors, {ivf.nlist} lists, code size {ivf.code_size}")

    ondisk = faiss.OnDiskInvertedLists(ivf.nlist, ivf.code_size, str(data_path))
    pointers = faiss.InvertedListsPtrVector()
    pointers.push_back(ivf.invlists)
    copied = ondisk.merge_from_multiple(pointers.data(), pointers.size(), False, True)
    note(f"copied {copied:,} entries into {data_path}")
    if copied != source.ntotal:
        raise SystemExit("Entry count mismatch.")

    target = faiss.read_index(str(args.source), faiss.IO_FLAG_MMAP)
    faiss.extract_index_ivf(target).replace_invlists(ondisk, True)
    ondisk.this.disown()
    faiss.write_index(target, str(index_path))
    note(f"wrote {index_path}")
    check(args.source, index_path, args.check_queries)


def check(source_path: Path, index_path: Path, count: int) -> None:
    """Same lists, so both indexes must return identical results."""
    source = faiss.read_index(str(source_path), faiss.IO_FLAG_MMAP)
    copy = faiss.read_index(str(index_path), faiss.IO_FLAG_READ_ONLY)
    if copy.ntotal != source.ntotal:
        raise SystemExit("Vector count mismatch.")
    rng = np.random.default_rng(0)
    queries = rng.standard_normal((count, source.d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for each in (source, copy):
        faiss.extract_index_ivf(each).nprobe = 64
    d0, i0 = source.search(queries, 128)
    d1, i1 = copy.search(queries, 128)
    if not (np.array_equal(i0, i1) and np.array_equal(d0, d1)):
        raise SystemExit("On-disk index results differ from the source index.")
    note(f"check passed: {count} queries identical")


if __name__ == "__main__":
    main()
