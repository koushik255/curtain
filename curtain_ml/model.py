"""Curtain's frame-embedding model."""

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.models import resnet18


class CurtainEncoder(nn.Module):
    """A ResNet-18 that produces normalized frame embeddings."""

    def __init__(self, embedding_dim: int = 128, image_size: int = 192):
        # Create a randomly initialized ResNet-18 projection head.
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Map a batch of images to unit-length embedding vectors.
        return functional.normalize(self.backbone(images), dim=1)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, device: str = "cpu"
    ) -> "CurtainEncoder":
        # Load model weights and architecture settings from a saved run.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(
            embedding_dim=int(config["embedding_dim"]),
            image_size=int(config["image_size"]),
        )
        model.load_state_dict(checkpoint["model"])
        return model.to(device).eval()
