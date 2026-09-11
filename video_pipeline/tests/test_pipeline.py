import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gpu_video_pipeline.decoder import SampledBatch
from gpu_video_pipeline.pipeline import VideoIndexer


class TinyEncoder(torch.nn.Module):
    def forward(self, images):
        means = images.mean(dim=(2, 3))
        return torch.cat((means, means[:, :1]), dim=1)


class FakeSampler:
    sample_fps = 2.0
    duration_seconds = 1.0
    source_fps = 24.0
    width = 8
    height = 6

    def __init__(self, movie):
        self.movie = movie
        self.frames = [
            torch.full((3, 6, 8), value, dtype=torch.uint8) for value in (32, 192)
        ]

    def __len__(self):
        return len(self.frames)

    def batches(self, batch_size):
        self.requested_batch_size = batch_size
        yield SampledBatch(self.frames, [0.0, 0.5], [0, 12])


class PipelineTests(unittest.TestCase):
    def test_cpu_pipeline_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            movie = root / "movie.mp4"
            movie.write_bytes(b"fake movie")
            output = root / "index"
            sampler = FakeSampler(movie)
            indexer = VideoIndexer(
                TinyEncoder(), image_size=4, embedding_dim=4,
                device="cpu", use_float16=False,
            )

            manifest = indexer.index(sampler, output, batch_size=2)

            matrix = np.load(output / "embeddings.npy")
            records = json.loads((output / "records.json").read_text())
            self.assertEqual(matrix.shape, (2, 4))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual([record["timestamp_seconds"] for record in records], [0.0, 0.5])
            self.assertEqual(manifest["frames"], 2)
            self.assertFalse((output / "embeddings.npy.partial").exists())


if __name__ == "__main__":
    unittest.main()
