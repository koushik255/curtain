from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
    "sscd_disc_mixup.torchscript.pt"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/sscd_disc_mixup.torchscript.pt"


def ensure_model(path: Path) -> Path:
    """Download the official SSCD checkpoint once."""
    path = path.expanduser().resolve()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading SSCD to {path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu instead.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use --device cpu instead.")
    return torch.device(requested)


class SSCDModel:
    """Turn images into normalized SSCD copy-detection descriptors."""

    dimensions = 512

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, device: str = "auto"):
        self.requested_device = device
        self.model_path = ensure_model(model_path)
        self.device = choose_device(device)
        self._load_model()

    def _load_model(self) -> None:
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.model = self.model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        print(f"SSCD is using {self.device.type}.")

    def _prepare_batch(self, images: list[Image.Image]) -> torch.Tensor:
        prepared: list[np.ndarray] = []
        for image in images:
            image = ImageOps.exif_transpose(image).convert("RGB")
            resized = image.resize((320, 320), Image.Resampling.BICUBIC)
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
            prepared.append(np.transpose(pixels, (2, 0, 1)))
        batch = torch.from_numpy(np.stack(prepared))
        return (batch - self.mean) / self.std

    def _encode_once(self, images: list[Image.Image]) -> np.ndarray:
        batch = self._prepare_batch(images).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(batch)
        values = embeddings.float().cpu().numpy()
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(lengths, 1e-12)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        try:
            return self._encode_once(images)
        except (RuntimeError, NotImplementedError) as error:
            if self.requested_device != "auto" or self.device.type == "cpu":
                raise
            print(f"{self.device.type.upper()} failed ({error}). Falling back to CPU.")
            self.device = torch.device("cpu")
            self._load_model()
            return self._encode_once(images)
