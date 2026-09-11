import unittest
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpu_video_pipeline.decoder import NvdecSampler, frame_to_chw, sample_timestamps


class DecoderTests(unittest.TestCase):
    def test_half_open_timeline(self):
        self.assertEqual(sample_timestamps(1.0, 2.0), [0.0, 0.5])
        self.assertEqual(sample_timestamps(1.01, 2.0), [0.0, 0.5, 1.0])

    def test_invalid_timeline(self):
        for duration, fps in [(0, 2), (-1, 2), (1, 0), (1, float("nan"))]:
            with self.subTest(duration=duration, fps=fps):
                with self.assertRaises(ValueError):
                    sample_timestamps(duration, fps)

    def test_packed_rgbp_becomes_chw(self):
        packed = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(12, 5)
        result = frame_to_chw(packed)
        self.assertEqual(result.shape, (3, 4, 5))
        self.assertEqual(result[1, 0, 0], packed[4, 0])

    def test_hwc_becomes_chw(self):
        hwc = torch.zeros((4, 5, 3), dtype=torch.uint8)
        self.assertEqual(frame_to_chw(hwc).shape, (3, 4, 5))

    def test_sampler_maps_times_and_batches_without_duplicate_source_frames(self):
        class FakeDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return SimpleNamespace(
                    duration=1.1, avg_frame_rate=24, width=1920, height=1080
                )

            def get_index_from_time_in_seconds(self, timestamp):
                return round(timestamp)

            def get_batch_frames_by_index(self, indexes):
                return list(indexes)

        fake_nvc = SimpleNamespace(
            SimpleDecoder=FakeDecoder,
            OutputColorType=SimpleNamespace(RGBP="rgbp"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            movie = Path(temporary) / "movie.mp4"
            movie.touch()
            with patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}):
                sampler = NvdecSampler(movie, sample_fps=2)

            batches = list(sampler.batches(1))
            self.assertEqual(len(sampler), 2)
            self.assertEqual([batch.source_indices for batch in batches], [[0], [1]])
            self.assertEqual([batch.timestamps for batch in batches], [[0.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
