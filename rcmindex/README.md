# rcmindex

C++/HIP overlapping-tile indexer for the ROCm (MI300X/MI325X) GPU droplet.
It writes the same layout as the 75k Python indexer: `OUT.npy` (fp16
`[frames, 49, 128]`), `OUT.records.json`, `OUT.json` (timing report), plus
`OUT.global.npy` (`[frames, 128]`) with `--global`.

Per frame it runs the ResNet-18 trunk **once** at 768x768 and pools the 49
half-stride 4x4 tiles from the 24x24 feature map, instead of 49 separate
192x192 passes. Stages overlap across rotating slots:

| Stage | Where |
|---|---|
| read JPEG bytes | parallel host reader threads |
| decode | rocJPEG hardware decoder, RGB planar in device memory |
| resize + normalize | fused antialiased HIP kernel → fp16 NCHW |
| trunk | MIGraphX compiled fp16 program (cached as `.mxr`) |
| pool + fc + L2 | one HIP kernel |
| write | pinned async copy, written while the next batch computes |

## 1. Export the model (once, any machine with the repo)

```sh
uv run --with onnx python -m rcmindex.export_model \
  --checkpoint .local/trained_models/combined-rocm-50k-v1/best.pt \
  --output .local/rcmindex-model/combined-rocm-50k-v1
```

Add `--check frame1.jpg frame2.jpg ...` to compare shared-trunk tiles with the
current per-tile embeddings on CPU. On 20 frames: mean cosine 0.91, p5 0.85, so
validate retrieval on the 75k benchmark before a full run.

## 2. Build on the droplet

Needs ROCm 7.x with `migraphx` and `rocjpeg` dev packages.

```sh
cmake -S rcmindex -B rcmindex/build -DCMAKE_BUILD_TYPE=Release
cmake --build rcmindex/build -j
```

## 3. Index

```sh
rcmindex/build/rcmindex \
  --frames /mnt/curtain/data/screenshots \
  --model /mnt/curtain/rcmindex-model/combined-rocm-50k-v1 \
  --output /mnt/curtain/indexes/overlap-full/curtain-overlap.npy \
  --batch 32 --global
```

The first run compiles the trunk with MIGraphX (minutes) and caches
`trunk_b<batch>_s<size>.mxr` in the model directory. Use `--limit 75000` to
reproduce the 75k benchmark (previous Python indexer: 525 frames/s).

Not done yet: int8, and hipGraph capture (MIGraphX programs can't be captured;
the remaining launches are only the two resize passes and the pool kernel).
