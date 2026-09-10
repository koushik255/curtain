import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from curtain_ml.index import CudaIndexer, JpegBytes, frame_number


class IndexingTests(unittest.TestCase):
    def test_frame_number(self):
        self.assertEqual(frame_number("frame_001234.jpg"), 1234)
        self.assertIsNone(frame_number("poster.jpg"))

    def test_dataset_reads_only_jpeg_bytes_in_filename_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            movie = Path(temporary)
            Image.new("RGB", (32, 18)).save(movie / "frame_000002.jpg")
            Image.new("RGB", (32, 18)).save(movie / "frame_000001.jpeg")
            (movie / "notes.txt").write_text("not a frame")

            dataset = JpegBytes(movie)

            self.assertEqual(
                [path.name for path in dataset.paths],
                ["frame_000001.jpeg", "frame_000002.jpg"],
            )
            self.assertEqual(dataset[0][:2], b"\xff\xd8")

    def test_indexer_refuses_to_run_without_cuda(self):
        with patch("curtain_ml.index.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                CudaIndexer(Path("checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
