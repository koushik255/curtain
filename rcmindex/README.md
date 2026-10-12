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

The single-directory mode still works:
`rcmindex --frames DIR --model DIR --output OUT.npy [--limit N] [--global]`.

Not done yet: int8, and hipGraph capture (MIGraphX programs can't be captured;
the remaining launches are only the two resize passes and the pool kernel).
