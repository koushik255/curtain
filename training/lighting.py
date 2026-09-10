"""On-the-fly RGB appearance augmentation; not physically accurate relighting."""
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class LightingConfig:
    clean_probability: float = 0.2
    mild_probability: float = 0.5
    brightness: tuple[float, float] = (0.6, 1.4)
    contrast: tuple[float, float] = (0.75, 1.25)
    saturation: tuple[float, float] = (0.85, 1.15)
    gamma: tuple[float, float] = (0.75, 1.35)
    temperature_strength: float = 0.08

    def __post_init__(self):
        if not (0 <= self.clean_probability <= 1 and 0 <= self.mild_probability <= 1
                and self.clean_probability + self.mild_probability <= 1):
            raise ValueError('Clean and mild probabilities must sum to at most one')
        for name in ('brightness', 'contrast', 'saturation', 'gamma'):
            low, high = getattr(self, name)
            if not 0 < low <= high < float('inf'):
                raise ValueError(f'Invalid {name} range')
        if not 0 <= self.temperature_strength <= 0.25:
            raise ValueError('Temperature strength must be between zero and 0.25')


class RandomLighting:
    """20% unchanged lighting, 50% mild, 30% broader by default.

    Uses Torch RNG so DataLoader worker seeds and torch.manual_seed apply.
    The clean branch bypasses lighting only, not later JPEG/resize transforms.
    """

    def __init__(self, config: LightingConfig | None = None):
        self.config = config or LightingConfig()

    @staticmethod
    def sample(bounds):
        return float(torch.empty(()).uniform_(*bounds))

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            raise ValueError('Lighting augmentation expects an RGB image')
        cfg = self.config
        draw = float(torch.rand(()))
        if draw < cfg.clean_probability:
            return image.copy()
        mild = draw < cfg.clean_probability + cfg.mild_probability
        def bounds(full):
            return tuple(1 + (value - 1) * 0.35 for value in full) if mild else full
        image = transforms.ColorJitter(brightness=bounds(cfg.brightness),
                                       contrast=bounds(cfg.contrast),
                                       saturation=bounds(cfg.saturation))(image)
        image = TF.adjust_gamma(image, self.sample(bounds(cfg.gamma)))
        strength = cfg.temperature_strength * (0.35 if mild else 1)
        shift = self.sample((-strength, strength))
        # Simple warm/cool channel gains, intentionally not a Kelvin simulation.
        channels = image.split()
        return Image.merge('RGB', tuple(channel.point(
            [min(255, max(0, round(value * gain))) for value in range(256)])
            for channel, gain in zip(channels, (1 + shift, 1., 1 - shift))))
