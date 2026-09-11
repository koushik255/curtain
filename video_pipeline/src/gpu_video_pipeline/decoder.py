"""GPU movie decoding and time-based frame sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SampledBatch:
    frames: Sequence[Any]
    timestamps: list[float]
    source_indices: list[int]


def sample_timestamps(duration_seconds: float, sample_fps: float) -> list[float]:
    """Return a constant-rate, half-open sampling timeline."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    count = max(1, math.ceil(duration_seconds * sample_fps - 1e-9))
    return [position / sample_fps for position in range(count)]


def frame_to_chw(frame: Any) -> torch.Tensor:
    """Expose a decoded RGB/RGBP frame as a CHW PyTorch tensor."""
    tensor = torch.from_dlpack(frame)
    if tensor.ndim == 2 and tensor.shape[0] % 3 == 0:
        tensor = tensor.reshape(3, tensor.shape[0] // 3, tensor.shape[1])
    elif tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise RuntimeError(f"Expected an RGB frame, received shape {tuple(tensor.shape)}")
    return tensor


class NvdecSampler:
    """Decode requested movie times directly into CUDA device memory."""

    def __init__(self, movie: Path, sample_fps: float = 2.0, gpu_id: int = 0):
        try:
            import PyNvVideoCodec as nvc
        except ImportError as error:
            raise RuntimeError("PyNvVideoCodec is required for GPU decoding") from error

        self.movie = movie.expanduser().resolve()
        if not self.movie.is_file():
            raise FileNotFoundError(self.movie)
        self.sample_fps = sample_fps
        self.decoder = nvc.SimpleDecoder(
            str(self.movie),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )
        metadata = self.decoder.get_stream_metadata()
        self.duration_seconds = float(metadata.duration)
        self.source_fps = float(metadata.average_fps)
        self.width = int(metadata.width)
        self.height = int(metadata.height)

        pairs = []
        for timestamp in sample_timestamps(self.duration_seconds, sample_fps):
            try:
                index = int(self.decoder.get_index_from_time_in_seconds(timestamp))
            except Exception:
                # Matroska containers can advertise a longer duration than the
                # decodable stream; the tail timestamps then raise. Stop here.
                break
            pairs.append((index, timestamp))
        self.samples: list[tuple[int, float]] = []
        seen = set()
        for pair in pairs:
            if pair[0] not in seen:
                self.samples.append(pair)
                seen.add(pair[0])

    def __len__(self) -> int:
        return len(self.samples)

    def batches(self, batch_size: int) -> Iterator[SampledBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            selected = self.samples[start : start + batch_size]
            indices = [index for index, _ in selected]
            frames = self.decoder.get_batch_frames_by_index(indices)
            if len(frames) != len(selected):
                raise RuntimeError(
                    f"Decoder returned {len(frames)} frames for {len(selected)} indexes"
                )
            yield SampledBatch(
                frames=frames,
                timestamps=[timestamp for _, timestamp in selected],
                source_indices=indices,
            )
