# Curtain

Curtain finds the movie frame a screenshot came from, even a tight crop of it.
The active code is small:

```text
curtain_ml/          Encoder, training and retrieval logic
cloud/               Modal training and indexing jobs
rcmindex/            C++/HIP tile indexer for AMD GPUs, FAISS search, alignment
apps/                Search site
benchmarks/          Retrieval robustness checks against the live index
deploy/              systemd unit for the site
```

The encoder is a ResNet-18 trained from scratch with contrastive learning. Two
altered views of the same frame form a positive pair; all other frames in the
batch are negatives. The current augmentation recipe changes crop, lighting,
resolution, blur, and JPEG quality. The resulting 128-value embeddings are
L2-normalized, so similarity is a dot product (cosine similarity).

Tight crops don't resemble their whole frame, so every frame is indexed as 49
overlapping tiles (a 4x4 grid stepped by half a tile) plus the whole frame.
The current model, `combined-rocm-50k-v1`, trains on full frames and tiles
together on an AMD MI325X ([docs/rocm-training.md](docs/rocm-training.md)).
[rcmindex](rcmindex/README.md) computes all 49 tile embeddings from one network
pass per frame with custom HIP kernels, at about 3,000 frames/s.

A search embeds the query, finds candidate frames in a FAISS index over all
tiles, re-scores them exactly, and checks the best few with SIFT alignment to
show where the screenshot sits in the frame.

Large inputs and local-only material live in the ignored `.local/` directory:

```text
.local/
  screenshots/             Source frames
  selected_screenshots*/   Training selections
  trained_models/          Checkpoints and experiment reports
  trained_indexes/         Frame embeddings
  rcmindex-model/          Exported trunk and head for rcmindex
  rcmindex-runs/           rcmindex tile arrays and timing reports
  faiss/                   FAISS tile index and catalog
  benchmarks/              Validation suite and results
  transfer/                Transfer archive
  output/                  Generated assets
  web/                     Local website copy
  archive/                 Retired Rust, video, scripts, tests, and probes
```

The standard Python environment remains at `.venv/` so `uv` and the installed
service continue to work; Git ignores it too.

## Train and index

Training and indexing run on GPU hosts. The current model comes from the ROCm
trainer ([docs/rocm-training.md](docs/rocm-training.md)), and its tile index
from `rcmindex` on an AMD GPU droplet, one command from this machine
([rcmindex/README.md](rcmindex/README.md)).

The original Modal jobs (whole-frame embeddings) are still here:

```sh
.venv/bin/modal run cloud/train.py --epochs 20 --run-name my-run
.venv/bin/modal run cloud/index.py --dry-run
.venv/bin/modal run cloud/index.py
```

See [docs/training.md](docs/training.md) for the data and artifact flow.

## Search site

The local server loads its checkpoint, FAISS index, tile arrays, and frames
from `.local/`:

```sh
uv run python -m apps.server --host 127.0.0.1 --port 8781
```

Cloudflare Tunnel exposes it publicly at
`https://curtain.koushikkoushik.com`. The origin remains bound to
`127.0.0.1:8781`; only the local tunnel connector can reach it.

The Rust extractor, NVIDIA video pipeline, old tests, and one-off probes are
preserved in `.local/archive/` but are no longer part of the Git-facing project.
