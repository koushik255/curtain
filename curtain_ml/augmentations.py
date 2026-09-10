"""Image augmentations used to train Curtain's embedding model."""

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class RandomFrameCrop:
    """Randomly change framing while retaining most of the source image."""

    def __init__(self, probability=0.7, minimum_fraction=0.75):
        # Configure how often and how tightly the frame is cropped.
        if not 0 <= probability <= 1 or not 0 < minimum_fraction <= 1:
            raise ValueError("Invalid crop probability or minimum fraction")
        self.probability = probability
        self.minimum_fraction = minimum_fraction

    def box(self, size):
        # Choose a valid crop rectangle, or the full image for identity.
        width, height = size
        if float(torch.rand(())) >= self.probability:
            return (0, 0, width, height)
        width_fraction = float(torch.empty(()).uniform_(self.minimum_fraction, 1))
        height_fraction = float(torch.empty(()).uniform_(self.minimum_fraction, 1))
        crop_width = max(1, round(width * width_fraction))
        crop_height = max(1, round(height * height_fraction))
        left = int(torch.randint(width - crop_width + 1, ()))
        top = int(torch.randint(height - crop_height + 1, ()))
        return (left, top, left + crop_width, top + crop_height)

    def __call__(self, image):
        # Return the image with the sampled framing change applied.
        return image.crop(self.box(image.size))


@dataclass(frozen=True)
class LightingConfig:
    """Probability and range settings for RandomLighting."""
    clean_probability: float = 0.2
    mild_probability: float = 0.5
    brightness: tuple[float, float] = (0.6, 1.4)
    contrast: tuple[float, float] = (0.75, 1.25)
    saturation: tuple[float, float] = (0.85, 1.15)
    gamma: tuple[float, float] = (0.75, 1.35)
    temperature_strength: float = 0.08

    def __post_init__(self):
        # Validate probability and appearance ranges after construction.
        probabilities_are_valid = (
            0 <= self.clean_probability <= 1
            and 0 <= self.mild_probability <= 1
            and self.clean_probability + self.mild_probability <= 1
        )
        if not probabilities_are_valid:
            raise ValueError("Clean and mild probabilities must sum to at most one")
        for name in ("brightness", "contrast", "saturation", "gamma"):
            low, high = getattr(self, name)
            if not 0 < low <= high < float("inf"):
                raise ValueError(f"Invalid {name} range")
        if not 0 <= self.temperature_strength <= 0.25:
            raise ValueError("Temperature strength must be between zero and 0.25")


class RandomLighting:
    """Apply clean, mild, or broad appearance variation using the Torch RNG."""

    def __init__(self, config: LightingConfig | None = None):
        # Use the supplied lighting recipe, or the project's default recipe.
        self.config = config or LightingConfig()

    @staticmethod
    def sample(bounds):
        # Sample a scalar with Torch's RNG so worker seeding stays reproducible.
        return float(torch.empty(()).uniform_(*bounds))

    def __call__(self, image: Image.Image) -> Image.Image:
        # Return a clean, mildly altered, or broadly altered RGB image.
        if image.mode != "RGB":
            raise ValueError("Lighting augmentation expects an RGB image")
        config = self.config
        draw = float(torch.rand(()))
        if draw < config.clean_probability:
            return image.copy()
        mild = draw < config.clean_probability + config.mild_probability

        def bounds(full):
            return tuple(1 + (value - 1) * 0.35 for value in full) if mild else full

        image = transforms.ColorJitter(
            brightness=bounds(config.brightness),
            contrast=bounds(config.contrast),
            saturation=bounds(config.saturation),
        )(image)
        image = TF.adjust_gamma(image, self.sample(bounds(config.gamma)))
        strength = config.temperature_strength * (0.35 if mild else 1)
        shift = self.sample((-strength, strength))
        channels = image.split()
        return Image.merge(
            "RGB",
            tuple(
                channel.point(
                    [min(255, max(0, round(value * gain))) for value in range(256)]
                )
                for channel, gain in zip(channels, (1 + shift, 1.0, 1 - shift))
            ),
        )
