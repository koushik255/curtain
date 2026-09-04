# curtain

A Rust command-line program that uses FFmpeg to save one JPEG screenshot every 0.5 seconds (2 frames per second) from movies. Screenshots are scaled down to a maximum width of 1280 pixels without enlarging smaller videos, and use moderate JPEG quality for faster processing and smaller files.

## Requirements

- Rust
- `ffmpeg` available on your `PATH`

## Process your Movies directory

Running without arguments scans `~/Downloads/Movies`:

```sh
cargo run --release
```

It processes movie files directly in that directory and files one directory deeper. It does not recurse further. Supported extensions include MP4, MKV, AVI, MOV, WebM, M4V, MPG, MPEG, WMV, and FLV.

Screenshots are placed in `~/curtain/screenshots/<movie-name>/`, whether the movie is directly in the Movies directory or in an immediate subdirectory:

```text
~/Downloads/Movies/film.mp4        -> ~/curtain/screenshots/film/frame_000001.jpg
~/Downloads/Movies/Action/film.mkv -> ~/curtain/screenshots/film/frame_000001.jpg
```

The terminal shows a live frame counter for each active movie and an overall completed-movies counter. Frames are written immediately. If the program is interrupted, running it again resumes each unfinished movie from its existing numbered JPEGs (and safely recreates the last frame in case it was only partially written).

Movies are processed two at a time by default using Rayon. Change the parallelism with `CURTAIN_JOBS`:

```sh
CURTAIN_JOBS=4 cargo run --release
```

FFmpeg already decodes each movie as one sequential pass, which is more efficient than launching a separate FFmpeg process for every timestamp.

## Check before running

Use `--check` to list every discovered movie and wait for confirmation:

```sh
cargo run --release -- --check
```

The first `--` is required because `--check` must be passed through Cargo to the program. Enter `y` or `yes` to proceed; any other answer cancels.

## Other paths

Scan another directory and optionally choose an output root:

```sh
cargo run --release -- /path/to/movies
cargo run --release -- --check /path/to/movies /path/to/output
```

A single movie is also supported:

```sh
cargo run --release -- movie.mp4
```

Build a standalone executable with:

```sh
cargo build --release
./target/release/curtain --check
```

## Build the movie-search index

The search side uses the same SSCD copy-detection model as `fig`, adapted for
movie frames. The extractor retains screenshots at 2 fps; the indexer uses every
other screenshot to create one embedding per second and groups search results by
movie. Indexing is resumable and each finished movie is kept separately, so
adding another movie does not rebuild the existing movies.

SSCD produces 32-bit descriptors, but Curtain stores them as 16-bit floats by
default and promotes them back to 32-bit for similarity calculations. Combined
with 1 fps sampling, this cuts the current collection's embedding storage from
roughly 1.1 GiB to about 280 MiB with negligible quantization loss. Pass
`--dtype float32` if full-precision storage is preferred.

Install the Python dependencies and build the index:

```sh
uv sync
uv run python -m src.index
```

The first run downloads the official `sscd_disc_mixup` checkpoint. On this
machine SSCD will use CUDA or MPS when available and otherwise use the CPU. The
indexer checkpoints after every batch, so it is safe to stop and rerun the same
command. Existing completed movies are skipped.

Useful indexing options include:

```sh
# Index just one matching movie folder
uv run python -m src.index --movie Matrix

# Tune memory use or accelerator throughput
uv run python -m src.index --batch-size 64

# A safe second worker can traverse from the opposite end
uv run python -m src.index --reverse

# Build a tiny disposable test index
uv run python -m src.index --limit 100 --output /tmp/curtain-test-index
```

Use `--force` only when a selected movie or incompatible partial index should be
rebuilt.

## Search

Search directly from the terminal:

```sh
uv run python -m src.search /path/to/query.jpg
```

The results show distinct movies, their strongest matching frame, the estimated
timestamp, and the SSCD similarity score.

## Current search pipeline

Curtain does not crop the images. The extractor first saves frames at 2 fps and
scales wide frames down to at most 1280 pixels. The indexer selects every other
saved frame, giving the current index a 1 fps sampling rate. Each selected frame
is converted to RGB, resized directly to 320 by 320 pixels, normalized, and sent
through `sscd_disc_mixup`. SSCD returns an L2-normalized 512-dimensional float32
descriptor, which Curtain stores as float16.

At query time Curtain performs the same resize and normalization, computes one
float32 SSCD descriptor, and compares it with all 286,967 stored descriptors by
dot product. Because the descriptors are normalized, this is cosine similarity.
The current implementation uses NumPy matrix multiplication over one shard per
movie; it does not currently use FAISS or an approximate-nearest-neighbor index.

## Experimental INT8 SSCD

The isolated experiment statically quantizes the SSCD network, calibrates it
with one test image per movie, and evaluates it with the other image. It does
not change the model used by the web service:

```sh
uv run python experiments/quantize_sscd.py
```

The generated experimental model is written under `models/`, which is ignored
by Git. The benchmark reports model size, FP32 and INT8 forward latency,
descriptor agreement, top-1 movie accuracy, and exact-frame agreement against
the existing index.

## Private web interface

Run the upload, drag-and-drop, and clipboard-paste interface locally with:

```sh
uv run python -m src.server
```

The persistent installation is available to this machine's Tailscale network at
`https://kouskous.tail90d2bb.ts.net:8443`. It uses a separate HTTPS port and
does not modify the existing StopAndGo route on port 443.
# curtain
