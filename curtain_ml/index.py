"""CUDA-only movie-frame indexing."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import IMAGENET_MEAN, IMAGENET_STD


BATCH_SIZE = 512
FILE_WORKERS = 8


class JpegBytes(Dataset):
    """Read compressed JPEGs; nvJPEG performs the actual decoding."""

    def __init__(self, movie_dir: Path):
        # The dataset stores filenames, not decoded images. This keeps the
        # constructor cheap and avoids putting every full-resolution frame in
        # memory at once.
        self.paths = sorted(
            path for path in movie_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if not self.paths:
            raise RuntimeError(f"No JPEG frames found in {movie_dir}.")

    def __len__(self) -> int:
        # Return the number of frames in the movie.
        return len(self.paths)

    def __getitem__(self, index: int) -> bytes:
        # DataLoader calls this method when it needs a particular frame.
        # At this point we only read the compressed file bytes; decoding is
        # deferred until preprocess() runs on the batch.
        return self.paths[index].read_bytes()


def preprocess(
    jpegs: list[bytes], image_size: int, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # Convert each compressed JPEG byte string into a 1-D uint8 tensor.
    # These tensors still contain compressed data; they are not images yet.
    encoded = [torch.frombuffer(bytearray(jpeg), dtype=torch.uint8) for jpeg in jpegs]

    # Decode the whole batch as RGB images directly on the CUDA device.
    # decode_jpeg returns one [3, height, width] tensor per JPEG.
    decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
    assert isinstance(decoded, list)  # decode_jpeg returns a list for list input

    # Stack separate frame tensors into one tensor shaped:
    # [batch, channels, height, width]. Stacking fails if source frames have
    # mismatched shapes. Convert straight to float16: inference runs in
    # float16 anyway, and skipping the float32 detour halves the memory
    # traffic on the full-resolution batch.
    try:
        images = torch.stack(decoded).to(torch.float16)
    except RuntimeError as error:
        raise RuntimeError("Frames within one movie must have the same dimensions.") from error

    # Resize every frame to the square input size expected by the model.
    images = functional.interpolate(
        images,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    # Apply the same ImageNet normalization used when the model was trained,
    # with /255, -mean, and /std folded into one fused multiply-add pass.
    # channels_last can improve convolution performance on CUDA.
    return images.mul_(scale).add_(bias).contiguous(memory_format=torch.channels_last)


def frame_number(path: Path | str) -> int | None:
    # Extract the numeric suffix from a frame filename when present.
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


class CudaIndexer:
    """Load one checkpoint and reuse it to index multiple movies on one GPU."""

    def __init__(self, checkpoint: Path):
        # Indexing requires a CUDA-capable GPU because decoding and model
        # inference are intentionally performed on CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError("Curtain indexing requires CUDA.")
        self.checkpoint = checkpoint
        self.model = CurtainEncoder.from_checkpoint(checkpoint, "cuda")
        self.model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

        # Precompute normalization constants once so preprocess() does not
        # re-upload them every batch. Folding x/255, -mean, and /std into
        # x * scale + bias lets normalization run as a single fused pass.
        mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
        self.scale = (1.0 / (255.0 * std)).to(torch.float16)
        self.bias = (-mean / std).to(torch.float16)

        # Run one small prediction before processing real data. This warms up
        # CUDA/cuDNN so one-time setup costs do not affect movie timings.
        sample = torch.zeros(
            (8, 3, self.model.image_size, self.model.image_size), device="cuda"
        ).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.model(sample)
        torch.cuda.synchronize()

    def index_movie(self, movie_dir: Path, output_dir: Path) -> dict:
        # Encode one movie into an atomic float16 matrix and metadata files.
        frames = JpegBytes(movie_dir)

        # DataLoader asks frames for individual indexes, groups the returned
        # JPEG byte strings into batches, and uses worker processes to read
        # files in parallel. It does not know anything about JPEGs itself;
        # it relies on JpegBytes.__len__() and JpegBytes.__getitem__().
        loader = DataLoader(
            frames,
            batch_size=BATCH_SIZE,
            num_workers=FILE_WORKERS,
            collate_fn=list,
            persistent_workers=True,
            prefetch_factor=2,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = output_dir / "embeddings.npy.partial"
        final = output_dir / "embeddings.npy"

        # Use a memory-mapped NumPy file so the complete embedding matrix does
        # not have to live in RAM. The partial suffix prevents unfinished work
        # from looking like a valid completed index.
        matrix = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.float16,
            shape=(len(frames), self.model.embedding_dim),
        )

        # Two pinned staging buffers let the device-to-host copy for batch N
        # be enqueued without blocking, so the CPU writes batch N-1 to the
        # memmap while the GPU is still working on batch N.
        buffers = [
            torch.empty(
                (BATCH_SIZE, self.model.embedding_dim),
                dtype=torch.float16,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        events = [torch.cuda.Event(), torch.cuda.Event()]
        pending: tuple[int, int, int] | None = None  # (buffer, row, count)

        def drain(matrix: np.memmap, entry: tuple[int, int, int]) -> None:
            # Wait for the copy into the pinned buffer to finish, then write
            # those rows into the memmap.
            buffer, row, count = entry
            events[buffer].synchronize()
            matrix[row : row + count] = buffers[buffer][:count].numpy()

        started = time.perf_counter()
        position = 0
        try:
            with torch.inference_mode():
                # Iteration starts here. Each pass gets one batch from the
                # DataLoader, which in turn calls JpegBytes.__getitem__() for
                # each frame index in that batch.
                for step, jpegs in enumerate(loader):
                    # Decode, resize, normalize, and move this batch to CUDA.
                    images = preprocess(jpegs, self.model.image_size, self.scale, self.bias)

                    # Run inference using float16 math to reduce GPU memory
                    # use and improve throughput. No gradients are recorded.
                    with torch.autocast("cuda", dtype=torch.float16):
                        embeddings = self.model(images)

                    # Enqueue an asynchronous copy of the embeddings into a
                    # pinned CPU buffer and record an event marking when it
                    # completes. The CPU does not wait here.
                    buffer = step % 2
                    count = embeddings.shape[0]
                    buffers[buffer][:count].copy_(embeddings, non_blocking=True)
                    events[buffer].record()

                    # Write the previous batch (whose copy has had a full
                    # batch of GPU work to finish) while this one computes.
                    if pending is not None:
                        drain(matrix, pending)
                    pending = (buffer, position, count)
                    position += count

                if pending is not None:
                    drain(matrix, pending)
                    pending = None

            # Ensure buffered memory-mapped data is written to disk.
            matrix.flush()
        except BaseException:
            # If decoding or inference fails, discard the incomplete index so
            # a later run can safely retry from the beginning.
            del matrix
            partial.unlink(missing_ok=True)
            raise

        # Release the memmap before checking/renaming the file.
        del matrix
        if position != len(frames):
            # A mismatch means some frames did not produce embeddings.
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Expected {len(frames)} embeddings, produced {position}.")

        # Publish the completed file only after every frame was processed.
        # Readers therefore see either the old complete index or the new
        # complete index, rather than a half-written file.
        os.replace(partial, final)

        records = []
        for path in frames.paths:
            number = frame_number(path)
            records.append({
                "movie": movie_dir.name,
                "filename": path.name,
                "frame_number": number,
                "timestamp_seconds": number / 2 if number is not None else None,
            })
        # Save the mapping from embedding row to original frame filename.
        (output_dir / "records.json").write_text(json.dumps(records) + "\n")
        seconds = time.perf_counter() - started
        manifest = {
            "movie": movie_dir.name,
            "frames": len(frames),
            "dimensions": self.model.embedding_dim,
            "dtype": "float16",
            "image_size": self.model.image_size,
            "checkpoint": self.checkpoint.name,
            "decoder": "nvjpeg",
            "batch_size": BATCH_SIZE,
            "workers": FILE_WORKERS,
            "encoding_seconds": seconds,
            "frames_per_second": len(frames) / seconds,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
        return manifest
