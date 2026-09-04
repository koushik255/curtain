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

Checkpoints and metrics persist under `/runs/<run-name>` in the
`curtain-training-data` Modal Volume.

## Build a complete movie index on Modal

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
