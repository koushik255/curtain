# rcmindex

C++ overlapping-tile indexer for ROCm GPUs (HIP + MIGraphX; MI300X/MI325X)
and NVIDIA GPUs (CUDA + TensorRT; the RTX 4060). Both builds share the same
kernels (`src/kernels.hip`) and pipeline; `src/gpu.hpp` maps the HIP runtime
calls to CUDA.
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

Both builds need `cmake` (3.25+), a C++20 compiler (GCC 13+), `libturbojpeg0-dev`
and `libwebp-dev`.

NVIDIA (the 4060 server): the CUDA toolkit (12.x, with `nvcc`) and TensorRT 10
(`libnvinfer-dev`, `libnvonnxparsers-dev`, or a tarball passed as `-DTENSORRT_ROOT=`).
`CMAKE_CUDA_ARCHITECTURES` defaults to 89 (RTX 40xx).

```sh
cmake -S rcmindex -B rcmindex/build-cuda -DRCM_BACKEND=cuda -DCMAKE_BUILD_TYPE=Release
cmake --build rcmindex/build-cuda -j"$(nproc)"
```

AMD: ROCm and MIGraphX (`migraphx-dev`).

```sh
cmake -S rcmindex -B rcmindex/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build rcmindex/build -j"$(nproc)"
```

The first run builds the trunk for each batch size and resolution and caches
it next to the model (`trunk_b32_s768.plan` for TensorRT, `.mxr` for
MIGraphX); a `.plan` only loads on the GPU and TensorRT version that built
it, so delete it after changing either. Batch 32 should fit the 4060's 8 GB;
pass `--batch 16` if it runs out of memory.

Frames are read as `.jpg`, or as `.webp` where only the 640 px copy is kept
(`.local/frames-small/<collection>/<movie>/`). Records always use the `.jpg`
name, so a movie indexed from its WebP copies matches the catalog. Each movie's
parent folder name becomes its collection: `<output>/<collection>/<movie>.npy`.

```sh
# One movie from its WebP copies (the parent folder is already its collection):
ls -d .local/frames-small/l4-50k-lbfive/The_Godfather_1972 | rcmindex/build-cuda/rcmindex \
  --model .local/rcmindex-model/combined-rocm-50k-v1 --movies-from - --global \
  --output .local/rcmindex-runs/$(date +%Y%m%d-%H%M%S)
```

A full re-index takes the six validation movies from their originals and
every other movie from its WebP copies:

```sh
mkdir -p .local/index-input/full-curtain
for d in .local/screenshots/*; do  # only movies with no WebP copy
  [ -d .local/frames-small/full-curtain/"${d##*/}" ] || ln -sfn "$PWD/$d" .local/index-input/full-curtain/
done
{ ls -d .local/index-input/full-curtain/*; ls -d .local/frames-small/*/*; } | rcmindex/build-cuda/rcmindex \
  --model .local/rcmindex-model/combined-rocm-50k-v1 --movies-from - --global \
  --output .local/rcmindex-runs/$(date +%Y%m%d-%H%M%S)
```

Check a new GPU or backend against the indexed arrays first (same frames and
model should agree to fp16 noise):

```sh
ls -d .local/index-input/full-curtain/* | rcmindex/build-cuda/rcmindex \
  --model .local/rcmindex-model/combined-rocm-50k-v1 --movies-from - --global --output .local/rcmindex-runs/cuda-check
.venv/bin/python -m rcmindex.compare_runs .local/rcmindex-runs/cuda-check
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
