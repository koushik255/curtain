# ROCm combined training

The combined trainer keeps JPEG decoding in parallel CPU workers and performs
all learned image transformations on the GPU. It generates four views per
source frame:

1. full frame with appearance variation;
2. full, mild-center, off-center, or 60–70% global crop;
3. exact 4x4 grid tile;
4. tile-sized crop shifted by up to half a tile.

Crop/resize is fused through `grid_sample`. Lighting, blur, resolution
degradation, codec-like quantization, and ImageNet normalization are batched
tensor operations. The objective averages independent global and local
contrastive losses so local views do not become false negatives for their own
full frames.

The trainer uses PyTorch's `cuda` device API. This is intentional: ROCm builds
expose AMD accelerators through the same API and identify the backend through
`torch.version.hip`.

On an AMD-ready machine, first verify the environment:

```sh
python - <<'PY'
import torch
print("available:", torch.cuda.is_available())
print("ROCm:", torch.version.hip)
print("device:", torch.cuda.get_device_name(0))
PY
```

Run a bounded throughput test before renting the machine for a full run:

```sh
python -m curtain_ml.gpu_training \
  --data /data/selected_screenshots \
  --output /data/runs/combined-rocm-smoke \
  --epochs 1 \
  --batch-size 1024 \
  --workers 4 \
  --forward-chunk-size 2048 \
  --limit 4096
```

Then run the full recipe:

```sh
python -m curtain_ml.gpu_training \
  --data /data/selected_screenshots \
  --output /data/runs/combined-rocm-v1 \
  --epochs 30 \
  --batch-size 1024 \
  --workers 4 \
  --forward-chunk-size 2048 \
  --evaluation-interval 2
```

These settings were tuned on a 256 GiB MI325X. Because every source produces
four model inputs, a source batch of 1,024 becomes 4,096 encoder inputs per
optimization step. They are forwarded in 2,048-image chunks to avoid MIOpen's
very-large-batch BatchNorm limit while retaining all 1,024 contrastive
negatives. Keep the loader worker count and prefetch depth conservative:
prefetched batches contain full-resolution decoded frames and can exhaust host
shared memory well before GPU memory is full.
