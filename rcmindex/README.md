# rcmindex

C++/HIP overlapping-tile indexer for the ROCm (MI300X/MI325X) GPU droplet.
It writes the same layout as the 75k Python indexer: `OUT.npy` (fp16
`[frames, 49, 128]`), `OUT.records.json` and `OUT.json` (timing report).

Per frame it runs the ResNet-18 trunk **once** at 768x768 and pools the 49
half-stride 4x4 tiles from the 24x24 feature map (6x6 windows, stride 3),
instead of 49 separate 192x192 passes. Stages overlap across rotating slots:

| Stage | Where |
|---|---|
| read + decode JPEG | 16 CPU reader threads, libjpeg-turbo, into pinned memory |
| upload | pinned RGB → device memory |
| resize + normalize | two antialiased HIP passes (horizontal, then vertical + normalize) → fp16 NCHW |
| trunk | MIGraphX compiled fp16 program (cached as `.mxr`) |
| pool + fc + L2 | one HIP kernel: one block per tile, one thread per channel |
| write | pinned async copy, written while the next batch computes |

JPEGs are decoded on the CPU: rocJPEG's VA-API driver loads a second LLVM
next to MIGraphX's and aborts the process.

Other tile grids come from the same trunk design: grid `g` runs the trunk at
`g*192` and pools `(2g-1)^2` tiles. `--grids 4,3,2,1` picks them; grid 4 writes
`OUT.npy`, grid 1 (the whole frame, also `--global`) writes `OUT.global.npy`,
and any other grid writes `OUT.g<g>.npy`.

On an MI325X, five movies with 49 tiles plus the whole frame ran at 2,750-3,050
frames/s (about 145k embeddings/s); a 21,433-frame movie took 7.0 s. The Python
indexer, which runs 49 separate tile passes, did 526 frames/s on its 75k run.

## 1. Export the model (once, any machine with the repo)

```sh
uv run --with onnx python -m rcmindex.export_model \
  --checkpoint .local/trained_models/combined-rocm-50k-v1/best.pt \
  --output .local/rcmindex-model/combined-rocm-50k-v1
```

Add `--check frame1.jpg frame2.jpg ...` to compare shared-trunk tiles with the
current per-tile embeddings on CPU. On 20 frames: mean cosine 0.91, p5 0.85, so
validate retrieval on the 75k benchmark before a full run.

## 2. Put the frames in Spaces (once, in the background)

```sh
python3 rcmindex/droplet/upload_frames.py            # all 157 movies, ~114 GB, resumable
python3 rcmindex/droplet/upload_frames.py --status
```

Streams each local movie folder (`.local/screenshots`, `~/lbfive/screenshots`)
as a tar into the private bucket `curtain-frames-koushik` (tor1), 2 at a time,
with progress and ETA. Stop it any time; a rerun skips stored movies. The
Spaces key is in `~/.config/curtain/spaces.env` and never leaves this machine.

## 3. Run on a GPU droplet (one command, from this machine)

```sh
python3 rcmindex/droplet/launch.py                   # 5 random movies from Spaces
python3 rcmindex/droplet/launch.py --movies 0        # every stored movie
python3 rcmindex/droplet/launch.py --source local --movies 2   # upload from here instead
python3 rcmindex/droplet/launch.py --keep / --destroy
```

It creates an MI325X (trying regions, then MI300X), copies `rcmindex/` and the
exported model, and starts the transfer as soon as ssh works, while the
droplet is still installing and building. From Spaces, the droplet downloads
8 movies at a time through presigned links (no key on the droplet); with
`--source local`, movies stream over `tar | ssh`. Each movie is marked
complete when it has arrived and `droplet/run_index.py` hands it to the
already-running `rcmindex` immediately. Transfer and GPU progress print in the
same terminal. Results are copied to `.local/rcmindex-runs/<time>/`, then the
droplet is deleted (also on failure or Ctrl-C).

Output per movie: `<collection>/<movie>.npy` (`[frames, 49, 128]` fp16),
`.global.npy`, `.records.json`, `.json` timing report.

`rcmindex` itself has two modes:

```sh
rcmindex --model DIR --frames DIR --output OUT.npy     # one directory
rcmindex --model DIR --movies-from - --output DIR      # movie dirs read from stdin
         [--grids 4,3,2,1] [--global] [--batch 32] [--slots 3] [--readers 16] [--limit N]
```

## 4. Build the search index

```sh
.venv/bin/python -m rcmindex.build_faiss
```

Puts every frame's 49 tiles and whole-frame vector into one FAISS index in
`.local/faiss/` (IVF with an HNSW coarse quantizer, PQ32, ~4 GB) plus
`catalog.json`. `rcmindex.search` takes FAISS candidates and re-scores them
exactly from the memory-mapped fp16 arrays; `rcmindex.align` then locates the
query inside the best frames with SIFT + RANSAC. The search site uses both.

Benchmarks: `benchmark_crops` (brute force over the tile arrays),
`benchmark_faiss` (the same queries through FAISS) and `validation_suite`
(eleven query types from the validation movies, every search system).

Not done yet: int8, and hipGraph capture (MIGraphX programs can't be captured;
the remaining launches are only the two resize passes and the pool kernels).
