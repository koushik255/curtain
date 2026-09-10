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

The trainer is CUDA-only. Modal extracts the existing 50,000-frame archive to
temporary local storage and persists the checkpoint under `/data/runs/<run>`:

```sh
.venv/bin/modal run cloud/train.py \
  --epochs 20 \
  --run-name l4-50k-crop-v2
```

The defaults—batch size 128, 192-pixel inputs, six validation movies, and eight
loader workers—live in `TrainingConfig` rather than being repeated throughout
the command line and Modal wrapper.

## Indexing

Training on 50,000 representative frames and indexing 2.48 million frames are
separate jobs. Indexing never modifies the model; it runs every searchable frame
through the frozen checkpoint once.

`cloud/index.py` prepares one task per movie and maps those tasks across at most
two L4 containers. Each container:

1. Loads and verifies the pinned checkpoint once.
2. Extracts one existing movie archive to temporary local disk.
3. Uses eight CPU workers only to read compressed JPEG bytes.
4. Uses nvJPEG on CUDA to decode them, then resizes and normalizes on the GPU.
5. Encodes batches of 512 and writes a preallocated float16 matrix.
6. Commits the movie index to the Modal Volume.

The two workers receive tasks from Modal's map queue, so a movie is handled by
exactly one worker. Completed, matching indexes are skipped on reruns.

Inspect the task list without renting a GPU:

```sh
.venv/bin/modal run cloud/index.py --dry-run
```

Run all movies and download both verified collections:

```sh
.venv/bin/modal run cloud/index.py
```

The active local outputs are:

```text
trained_models/l4-50k-crop-v1-20260909/best.pt
trained_indexes/l4-50k-crop-v1/
trained_indexes/l4-50k-lbfive-crop-v1/
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

The live site uses CPU inference because it is a persistent local service; the
large one-time training and indexing workloads use Modal CUDA GPUs.

Run the one retained robustness benchmark against the live artifacts with:

```sh
.venv/bin/python -m benchmarks.evaluate --condition crop --samples 100
```
