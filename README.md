# Curtain

Curtain finds the closest indexed movie frame to a supplied image. The active
code is small:

```text
curtain_ml/          Encoder and retrieval logic
cloud/               Training and indexing jobs
apps/                Search site
```

The encoder is a ResNet-18 trained from scratch with contrastive learning. Two
altered views of the same frame form a positive pair; all other frames in the
batch are negatives. The current augmentation recipe changes crop, lighting,
resolution, blur, and JPEG quality. The resulting 128-value embeddings are
L2-normalized, so search is a matrix-vector dot product (cosine similarity).

Large inputs and local-only material live in the ignored `.local/` directory:

```text
.local/
  screenshots/             Source frames
  selected_screenshots*/   Training selections
  trained_models/          Checkpoints and experiment reports
  trained_indexes/         Frame embeddings
  transfer/                Transfer archive
  output/                  Generated assets
  web/                     Local website copy
  archive/                 Retired Rust, video, scripts, tests, and probes
```

The standard Python environment remains at `.venv/` so `uv` and the installed
service continue to work; Git ignores it too.

## Train and index

Training and indexing run on GPU hosts:

```sh
.venv/bin/modal run cloud/train.py --epochs 20 --run-name my-run
.venv/bin/modal run cloud/index.py --dry-run
.venv/bin/modal run cloud/index.py
```

See [docs/training.md](docs/training.md) for the data and artifact flow.

## Search site

The local server loads its checkpoint, indexes, and frames from `.local/`:

```sh
uv run python -m apps.server --host 127.0.0.1 --port 8781
```

Cloudflare Tunnel exposes it publicly at
`https://curtain.koushikkoushik.com`. The origin remains bound to
`127.0.0.1:8781`; only the local tunnel connector can reach it.

The Rust extractor, NVIDIA video pipeline, old tests, and one-off probes are
preserved in `.local/archive/` but are no longer part of the Git-facing project.
