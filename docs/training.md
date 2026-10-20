# Training and indexing

## Training

The selected training archive contains one folder per movie. Complete movies
are assigned to either training or validation, preventing near-identical frames
from the same movie from leaking across the split.

For each training frame:

1. The anchor receives lighting, blur, resolution, and JPEG changes.
2. The positive receives those changes plus a random zoom/crop.
3. ResNet-18 converts both images into normalized 128-dimensional embeddings.
4. InfoNCE pulls the positive pair together and pushes other frames in the
   batch apart.
5. Validation measures exact-frame and movie top-1 retrieval after each epoch.
6. `best.pt` is saved whenever exact-frame validation accuracy improves.

The trainer needs a CUDA or ROCm GPU. It reads one folder per movie and writes
the checkpoint to the output directory:

```sh
.venv/bin/python -m curtain_ml.training \
  --data .local/selected_screenshots \
  --output .local/trained_models/l4-50k-crop-v2 \
  --epochs 20
```

The defaults (192-pixel inputs, six validation movies, eight loader workers)
live in `TrainingConfig` rather than on the command line.

## Indexing

Training on 50,000 representative frames and indexing 2.48 million frames are
separate jobs. Indexing never modifies the model; it runs every searchable frame
through the frozen checkpoint once.

These whole-frame indexes were built with `curtain_ml/index.py`
(`CudaIndexer`: CPU workers read JPEG bytes, nvJPEG decodes on the GPU, batches
of 512 are encoded into a preallocated float16 matrix). The tile indexer
[rcmindex](../rcmindex/README.md) has since replaced it.

The active local outputs are:

```text
.local/trained_models/l4-50k-crop-v1-20260909/best.pt
.local/trained_indexes/l4-50k-crop-v1/
.local/trained_indexes/l4-50k-lbfive-crop-v1/
```

Every collection and movie manifest stores the checkpoint hash, source hash,
frame count, and embedding shape. The web app refuses to combine artifacts that
do not match.

## Search

At startup, `apps/server.py` validates both collections and concatenates their
normalized matrices in memory. For a query it:

1. Decodes and normalizes the image.
2. Produces one embedding with the same checkpoint.
3. Multiplies chunks of the frame matrix by that embedding.
4. Streams progress and returns the highest cosine similarity.

The live site uses CPU inference because it is a persistent local service;
training and indexing need a GPU.

Run the one retained robustness benchmark against the live artifacts with:

```sh
.venv/bin/python -m benchmarks.evaluate --condition crop --samples 100
```
