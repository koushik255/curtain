# rcmindex

C++/HIP overlapping-tile indexer for ROCm GPUs (built for MI300X/MI325X).
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

## 2. Compute the tile embeddings

Build on a machine with ROCm and MIGraphX (`migraphx-dev`, `libturbojpeg0-dev`,
`cmake`), then feed it movie folders:

```sh
cmake -S rcmindex -B rcmindex/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build rcmindex/build -j"$(nproc)"
# Each movie's parent folder name becomes its collection: <output>/<collection>/<movie>.npy
mkdir -p .local/index-input/full-curtain
ln -s "$PWD/.local/screenshots/New_Movie_2024" .local/index-input/full-curtain/
ls -d .local/index-input/*/* | rcmindex/build/rcmindex \
  --model .local/rcmindex-model/combined-rocm-50k-v1 --movies-from - --global \
  --output .local/rcmindex-runs/$(date +%Y%m%d-%H%M%S)
```

Output per movie: `<collection>/<movie>.npy` (`[frames, 49, 128]` fp16),
`.global.npy`, `.records.json`, `.json` timing report.

`rcmindex` itself has two modes:

```sh
rcmindex --model DIR --frames DIR --output OUT.npy     # one directory
rcmindex --model DIR --movies-from - --output DIR      # movie dirs read from stdin
         [--grids 4,3,2,1] [--global] [--batch 32] [--slots 3] [--readers 16] [--limit N]
```

## 3. Build the search index

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
