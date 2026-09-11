# GPU video pipeline

This is an isolated experiment for indexing a movie without first creating a
directory of screenshots. It does not import from or modify Curtain.

```text
movie file
  -> time-based frame sampling
  -> NVIDIA NVDEC into CUDA RGB memory
  -> zero-copy DLPack views in PyTorch
  -> resize and normalize on CUDA
  -> user-supplied TorchScript encoder
  -> embeddings.npy + records.json + manifest.json
```

## Setup

The decoder requires an NVIDIA GPU, a compatible driver, and the video driver
capability inside containers.

```sh
cd video_pipeline
uv sync
```

## Run

The experiment deliberately accepts a standalone TorchScript encoder rather
than importing Curtain's model. The model must accept an NCHW float image batch
and return an `N x embedding_dim` tensor.

```sh
uv run gpu-video-index /path/to/movie.mkv \
  --model /path/to/encoder.ts \
  --image-size 192 \
  --embedding-dim 128 \
  --output indexes/movie
```

Frames are sampled at 2 fps by default. The output is written atomically: an
interrupted run leaves `embeddings.npy.partial`, never a completed matrix.

## Test

The unit tests exercise timeline construction, decoder batching, RGB-planar
conversion, and artifact generation without requiring a GPU:

```sh
uv run python -m unittest discover -s tests
```

The next step before any Curtain integration is a GPU benchmark comparing this
pipeline against its existing screenshot indexer for frame agreement,
embeddings per second, peak VRAM, bytes read, and output equivalence.
