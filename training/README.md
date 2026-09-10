# Curtain contrastive training

The training set is expected to contain one subdirectory per movie. Each source
frame produces two independently augmented views at load time. A randomly
initialized ResNet-18 maps them to normalized 128-dimensional embeddings, and an
in-batch InfoNCE loss pulls matching views together while pushing other frames
apart.

The default split holds six complete movies out for evaluation. The split is
deterministic and is recorded with every run.

## Local smoke test

```sh
uv run python -m training.train \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --limit 32 \
  --output /tmp/curtain-training-smoke
```

## Modal L4

Upload the selected frames once:

```sh
modal volume create curtain-training-data
modal volume put curtain-training-data selected_screenshots /selected_screenshots
```

Run a short GPU check:

```sh
modal run training/modal_train.py \
  --epochs 5 \
  --limit 512 \
  --run-name l4-smoke
```

Then run the full initial experiment:

```sh
modal run --detach training/modal_train.py \
  --epochs 20 \
  --run-name l4-baseline
```

For the balanced 50,000-frame dataset, select the frames locally and upload the
directory once. The L4 still loads only one training batch into GPU memory at a
time; the dataset does not need to be split into 5,000-frame jobs.

```sh
uv run python -m training.select_frames --total 50000
tar -cf /tmp/curtain-selected-screenshots-50k.tar -C selected_screenshots_50k .
modal volume put curtain-training-data \
  /tmp/curtain-selected-screenshots-50k.tar \
  /datasets/curtain-selected-screenshots-50k.tar
modal run training/modal_train.py \
  --epochs 20 \
  --archive-name curtain-selected-screenshots-50k.tar \
  --run-name l4-50k
```

Checkpoints and metrics persist under `/runs/<run-name>` in the
`curtain-training-data` Modal Volume.

## Build a complete movie index on Modal

To index every populated movie in `/home/koushik/curtain/screenshots` with the
frozen 50k checkpoint, use the project environment (the standalone Modal CLI
environment may lack NumPy):

```sh
.venv/bin/modal run training/modal_index_collection.py
```

This uploads one full-resolution movie archive at a time to
`/datasets/l4-50k` on the `curtain-training-data` Volume, extracts it onto
an L4's temporary local disk, and encodes batches of 512 frames. Each movie's
index is committed under `/indexes/l4-50k` and downloaded to
`trained_indexes/l4-50k/<movie>`. Local `collection.json` records completed movies.
Rerunning skips verified local results for the pinned checkpoint. All source
screenshots are included; the existing website and earlier indexes are separate.

To index the separate lbfive library without overwriting the Curtain collection:

```sh
.venv/bin/modal run training/modal_index_collection.py \
  --source /home/koushik/lbfive/screenshots --collection l4-50k-lbfive
```

This uses separate `/datasets/l4-50k-lbfive` and `/indexes/l4-50k-lbfive`
Volume paths and downloads to `trained_indexes/l4-50k-lbfive`. The checkpoint
is unchanged. It does not automatically change the website's active index.
The background run launched on September 7, 2026 can be inspected with
`journalctl --user -u curtain-index-lbfive -n 50 --no-pager`.

Upload a movie folder, encode it with the best checkpoint, and download the
compact index:

```sh
modal volume put curtain-training-data \
  screenshots/The_Matrix_1999 \
  /full_movies/The_Matrix_1999

modal run training/modal_index.py --movie-name The_Matrix_1999

modal volume get curtain-training-data \
  /indexes/l4-baseline/The_Matrix_1999 \
  trained_indexes/
```

Search the downloaded index locally with any query image:

```sh
uv run python -m training.search_trained screenshots/The_Matrix_1999/frame_000001.jpg \
  --checkpoint trained_models/curtain-resnet18-best.pt \
  --index trained_indexes/The_Matrix_1999 \
  --top-k 10
```

The command embeds the query with the same trained network and ranks the stored,
normalized movie embeddings by cosine similarity (implemented as a dot product).

## Private trained-model web interface

Run the separate trained-model site locally with:

```sh
uv run python -m training.server --host 127.0.0.1 --port 8781
```

Its persistent Tailnet installation is served separately at
`https://kouskous.tail90d2bb.ts.net:8444`. The existing services on ports 443
and 8443 are unchanged.

The site defaults to `trained_models/curtain-resnet18-50k-best.pt` and both full
`trained_indexes/l4-50k` and `trained_indexes/l4-50k-lbfive` collections. The extra
collection loads automatically when the default primary index is selected.
Startup checks each movie's checkpoint hash
and frame count before accepting searches. The Movies menu lists the indexed
films and links to their galleries; `/api/movies` exposes the same catalog.

The menu, galleries and search-result captions label movies as training,
validation, unseen by training/validation, or unknown. Provenance is loaded only
from a saved split whose sibling checkpoint matches the active checkpoint hash.
Movie names are matched case-insensitively. Shared titles are grouped while all
indexed frame rows are retained, with image paths resolved against their original
source roots. All 2,481,255 frames and 150 titles are searchable: 32 training,
6 validation and 112 unseen titles. The gallery supports all three groups.
Unseen here means absent
from the saved split, not a guarantee of no later diagnostic exposure. Labels
are movie-level, not per-frame membership in the training dataset.

## Local retrieval diagnostics

The crop-aware comparison runs with
`.venv/bin/modal run training/modal_benchmark_crops.py`. It compares original,
lighting-v1 and crop-v1 checkpoints with freshly encoded references for each.
The reference pool combines the selected 50k archive and all Lawrence of Arabia
frames. Eight fixed conditions cover original, center/off-center crops, tighter
zoom, aspect changes, and combined lighting/compression. The supplied difficult
Lawrence query and its successful control are evaluated separately, not trained on.
Results: `trained_models/crop-v1-broad-20260909.json` and Modal Volume
`/comparisons/crop-v1-broad-20260909.json`. The 768 synthetic source queries still
come from checkpoint-selection validation movies. This is not the full production
pool, nor a claim of an untouched final test set. Repeated trials on the Lawrence
example make it a development diagnostic.

For a paired L4 comparison of the original 50k and lighting-v1 checkpoints:

```sh
.venv/bin/modal run training/modal_benchmark_lighting.py
```

This uses the existing 50k archive, rebuilds separate normalized reference
vectors for each model, and evaluates 128 fixed-seed queries from each of the
six validation movies under 11 deterministic appearance conditions. It checks
the saved splits match. Results include per-query predictions and top-1/top-5
accuracy in `trained_models/lighting-v1-broad-20260909.json` and the Volume's
`/comparisons/lighting-v1-broad-20260909.json`. It does not modify production
indexes. These are checkpoint-selection validation movies, not an untouched
test set, and the 50k reference pool differs from the full 2.48M-frame benchmark.

### Lighting training experiment (opt-in)

The opt-in `--lighting-preset crop-v1` adds framing augmentation on top of
`lighting-v1`. Positive pairs keep one full-frame anchor and independently
augment the other view with a 70% chance of cropping. Each dimension retains
75–100% of the source, with independent width/height retention and random
position, before the normal resize. The remaining 30% retains full framing.
This limits crops to at least approximately 56% of the source area and avoids
nonoverlapping crop/crop pairs. Validation and default legacy behavior remain
unchanged. This is not a hard-negative-mining change.

The crop experiment uses the same 50k archive, 20 epochs, batch size 128,
192px inputs and one L4, initialized from scratch. Run name:
`l4-50k-crop-v1-20260909`. Background service:
`curtain-train-crop-v1-20260909.service`. Lawrence of Arabia and the supplied
query remain outside training. The known example is a development diagnostic,
not an untouched final test; evaluate multiple independent movies before deployment.
No automatic deployment or reindexing is performed.

```sh
.venv/bin/python -m unittest training.test_framing training.test_lighting training.test_scoring
```

`training/lighting.py` provides `LightingConfig` and `RandomLighting` using
Torchvision/PIL and the Torch worker RNG. `lighting-v1` selects 20% unchanged
lighting, 50% mild lighting, and 30% broader lighting per view. Broader ranges
are brightness 0.6–1.4, contrast 0.75–1.25, saturation 0.85–1.15, gamma 0.75–1.35,
and warm/cool red/blue gains up to 8%. Mild ranges are 35% as wide around identity.
Channel gains approximate color balance, not physical relighting. The unchanged
lighting branch still permits the existing blur/JPEG/resolution augmentations.

Use `--lighting-preset lighting-v1` with either `python -m training.train` or
`modal run training/modal_train.py`. The default `legacy` recipe is unchanged,
and the selected preset is recorded in config/checkpoints. Validation retains
the legacy query transforms for comparison; broader robustness must also be
evaluated separately on fixed held-out queries. No new Modal run is launched
by selecting or testing the module locally.

Use a new output directory / Modal run name for this experiment to preserve
existing checkpoints. The current trainer initializes a new model; this option
does not by itself enable fine-tuning from the 50k checkpoint.

The September 9 lighting experiment was launched with:

```sh
.venv/bin/modal run training/modal_train.py \
  --archive-name curtain-selected-screenshots-50k.tar \
  --run-name l4-50k-lighting-v1-20260909 --lighting-preset lighting-v1 \
  --epochs 20 --batch-size 128 --image-size 192 --workers 8
```

Its background service is `curtain-train-lighting-v1-20260909.service`; inspect
with `journalctl --user -u curtain-train-lighting-v1-20260909 -n 30 --no-pager`.
Modal app: `ap-7Y16WYeoXtcD0aQbiJZVvV`. Outputs use the separate Volume directory
`/runs/l4-50k-lighting-v1-20260909`. This run trains from scratch on the same
dataset/split as the previous 50k experiment. It does not deploy a model or
rebuild the search index automatically. Do not reuse the run name for a new run.

```sh
.venv/bin/python -m unittest training.test_lighting training.test_scoring
```

Run the fixed-seed, 314-source-frame diagnostic against both full collections:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -m training.test_collection_retrieval --normalize
.venv/bin/python -m training.audit_retrieval_failures
.venv/bin/python -m unittest training.test_scoring
```

Reports are saved under `trained_models/collection-robustness-20260908-normalized.json`
and `trained_models/retrieval-failure-audit-20260908.json`. The original unnormalized
report is preserved. This is a diagnostic sample, not a held-out generalization
benchmark: some movies were used in training and duplicate source frames remain
in the reference pool. Strict index-row accuracy counts identical copies as misses.

Search now re-normalizes float16 stored embeddings in float32 memory, without
changing the model or files on disk. On this sample, normalization improved dark
query top-1 from 94.9% to 95.5%, JPEG from 98.1% to 98.4%, and resized from 98.4%
to 99.0%; combined changes remained at 92.0%. Three of four normalized original
query misses returned pixel-identical copies; the other source was almost black.
The next training experiment should address lighting/combined degradation and
use a separate movie-disjoint validation/test set rather than tune to this sample.
