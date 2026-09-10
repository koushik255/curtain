# Curtain

Curtain finds the closest indexed movie frame to a supplied image. It has four
parts:

```text
src/main.rs          Extract movie frames with FFmpeg
curtain_ml/          Train the encoder and build indexes
cloud/               Run training and indexing on Modal L4 GPUs
apps/                Serve the private search site
```

The encoder is a ResNet-18 trained from scratch with contrastive learning. Two
altered views of the same frame form a positive pair; all other frames in the
batch are negatives. The current augmentation recipe changes crop, lighting,
resolution, blur, and JPEG quality. The resulting 128-value embeddings are
L2-normalized, so search is a matrix-vector dot product (cosine similarity).

Large inputs and outputs are intentionally ignored by Git: `screenshots/`,
`selected_screenshots*`, `trained_models/`, and `trained_indexes/`.

## Extract frames

The Rust extractor samples every movie at 2 fps, scales frames to at most 1280
pixels wide, and writes JPEGs to `screenshots/<movie>/`:

```sh
cargo run --release -- --check /path/to/movies /home/koushik/curtain/screenshots
```

It scans the given directory and its immediate subdirectories. Two movies run
in parallel by default; set `CURTAIN_JOBS` to change that. Existing frame
sequences are resumed.

## Train and index

Install the Python environment and run tests:

```sh
uv sync
uv run python -m unittest discover -s tests
```

Training and indexing intentionally require CUDA. They run on Modal L4s:

```sh
.venv/bin/modal run cloud/train.py --epochs 20 --run-name my-run
.venv/bin/modal run cloud/index.py --dry-run
.venv/bin/modal run cloud/index.py
```

See [docs/training.md](docs/training.md) for the data and artifact flow.

## Search site

The local server loads the crop-v1 checkpoint and both verified index
collections:

```sh
uv run python -m apps.server --host 127.0.0.1 --port 8781
```

Tailscale Serve exposes it privately on the tailnet at port 8444. The retired
8443 service is not part of this repository.
