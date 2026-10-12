"""Batched tensor augmentations for CUDA and ROCm training devices."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from curtain_ml.training import IMAGENET_MEAN, IMAGENET_STD


class GpuCombinedAugmenter:
    """Create global and local positive pairs entirely on an accelerator."""

    def __init__(self, image_size: int = 192, grid_size: int = 4):
        if image_size < 1:
            raise ValueError("Image size must be positive")
        if grid_size < 2:
            raise ValueError("Grid size must be at least two")
        self.image_size = image_size
        self.grid_size = grid_size

    @staticmethod
    def _uniform(
        count: int,
        low: float,
        high: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(count, device=device).uniform_(low, high)

    def _padded_batch(
        self, images: list[torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not images:
            raise ValueError("Cannot augment an empty image batch")
        if any(image.ndim != 3 or image.shape[0] != 3 for image in images):
            raise ValueError("Expected RGB tensors shaped [3, height, width]")
        heights = torch.tensor(
            [image.shape[1] for image in images], device=device, dtype=torch.float32
        )
        widths = torch.tensor(
            [image.shape[2] for image in images], device=device, dtype=torch.float32
        )
        maximum_height = int(heights.max().item())
        maximum_width = int(widths.max().item())
        batch = torch.zeros(
            (len(images), 3, maximum_height, maximum_width),
            device=device,
            dtype=torch.uint8,
        )
        for index, image in enumerate(images):
            height, width = image.shape[1:]
            batch[index, :, :height, :width].copy_(image, non_blocking=True)
        working_dtype = torch.float16 if device.type == "cuda" else torch.float32
        return batch.to(working_dtype), heights, widths

    @staticmethod
    def _full_boxes(heights: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(widths)
        return torch.stack((zeros, zeros, widths, heights), dim=1)

    def _global_boxes(
        self, heights: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """Mix full, mild-center, off-center, and strong global crops."""
        device = heights.device
        count = len(heights)
        draw = torch.rand(count, device=device)
        fractions = torch.ones(count, device=device)

        mild = (draw >= 0.20) & (draw < 0.45)
        off_center = (draw >= 0.45) & (draw < 0.75)
        strong = draw >= 0.75
        fractions[mild] = self._uniform(int(mild.sum()), 0.85, 1.0, device)
        fractions[off_center] = self._uniform(
            int(off_center.sum()), 0.70, 0.90, device
        )
        fractions[strong] = self._uniform(int(strong.sum()), 0.60, 0.70, device)

        crop_widths = widths * fractions
        crop_heights = heights * fractions
        available_x = widths - crop_widths
        available_y = heights - crop_heights
        random_x = torch.rand(count, device=device)
        random_y = torch.rand(count, device=device)
        left = available_x * random_x
        top = available_y * random_y

        # Mild crops stay near the center; other crop families cover the frame.
        center_jitter_x = self._uniform(count, -0.10, 0.10, device) * available_x
        center_jitter_y = self._uniform(count, -0.10, 0.10, device) * available_y
        centered_left = (available_x * 0.5 + center_jitter_x).clamp(min=0)
        centered_top = (available_y * 0.5 + center_jitter_y).clamp(min=0)
        centered_left = torch.minimum(centered_left, available_x)
        centered_top = torch.minimum(centered_top, available_y)
        left = torch.where(mild, centered_left, left)
        top = torch.where(mild, centered_top, top)
        return torch.stack(
            (left, top, left + crop_widths, top + crop_heights), dim=1
        )

    def _tile_boxes(
        self, heights: torch.Tensor, widths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact grid cells and same-sized crops shifted by half a tile."""
        device = heights.device
        count = len(heights)
        tile = torch.randint(self.grid_size * self.grid_size, (count,), device=device)
        rows = torch.div(tile, self.grid_size, rounding_mode="floor")
        columns = tile.remainder(self.grid_size)
        tile_widths = widths / self.grid_size
        tile_heights = heights / self.grid_size
        left = columns * tile_widths
        top = rows * tile_heights
        exact = torch.stack(
            (left, top, left + tile_widths, top + tile_heights), dim=1
        )

        delta_x = self._uniform(count, -0.5, 0.5, device) * tile_widths
        delta_y = self._uniform(count, -0.5, 0.5, device) * tile_heights
        shifted_left = (left + delta_x).clamp(min=0)
        shifted_top = (top + delta_y).clamp(min=0)
        shifted_left = torch.minimum(shifted_left, widths - tile_widths)
        shifted_top = torch.minimum(shifted_top, heights - tile_heights)
        shifted = torch.stack(
            (
                shifted_left,
                shifted_top,
                shifted_left + tile_widths,
                shifted_top + tile_heights,
            ),
            dim=1,
        )
        return exact, shifted

    def _crop_resize(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Fuse arbitrary rectangular crops and resize through grid sampling."""
        count, _channels, padded_height, padded_width = images.shape
        device = images.device
        dtype = images.dtype
        coordinates = torch.linspace(
            0, 1, self.image_size, device=device, dtype=dtype
        )
        x0, y0, x1, y1 = boxes.to(dtype).unbind(dim=1)
        x = x0[:, None, None] + (x1 - x0 - 1)[:, None, None] * coordinates[
            None, None, :
        ]
        y = y0[:, None, None] + (y1 - y0 - 1)[:, None, None] * coordinates[
            None, :, None
        ]
        x = x.expand(count, self.image_size, self.image_size)
        y = y.expand(count, self.image_size, self.image_size)
        normalized_x = x.mul(2 / max(1, padded_width - 1)).sub(1)
        normalized_y = y.mul(2 / max(1, padded_height - 1)).sub(1)
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        return functional.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    @staticmethod
    def _blur(images: torch.Tensor) -> torch.Tensor:
        kernel = images.new_tensor(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ).div_(16)
        kernel = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)
        padded = functional.pad(images, (1, 1, 1, 1), mode="reflect")
        return functional.conv2d(padded, kernel, groups=3)

    def _appearance(self, images: torch.Tensor) -> torch.Tensor:
        """Apply lighting, blur, degradation, quantization, and normalization."""
        device = images.device
        count = len(images)
        images = images.div(255).clamp_(0, 1)
        regime = torch.rand(count, device=device)
        clean = regime < 0.20
        mild = (regime >= 0.20) & (regime < 0.70)
        strength = torch.where(mild, 0.35, torch.ones_like(regime))
        strength = torch.where(clean, torch.zeros_like(strength), strength)

        def factor(low: float, high: float) -> torch.Tensor:
            sampled = self._uniform(count, low, high, device)
            return 1 + (sampled - 1) * strength

        brightness = factor(0.60, 1.40)
        contrast = factor(0.75, 1.25)
        saturation = factor(0.85, 1.15)
        gamma = factor(0.75, 1.35)
        temperature = self._uniform(count, -0.08, 0.08, device) * strength

        images = images * brightness[:, None, None, None]
        contrast_mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - contrast_mean) * contrast[:, None, None, None] + contrast_mean
        gray = (
            images[:, 0:1] * 0.299
            + images[:, 1:2] * 0.587
            + images[:, 2:3] * 0.114
        )
        images = gray + (images - gray) * saturation[:, None, None, None]
        images = images.clamp_(1e-4, 1).pow_(gamma[:, None, None, None])
        channel_gains = torch.stack(
            (1 + temperature, torch.ones_like(temperature), 1 - temperature), dim=1
        )
        images = (images * channel_gains[:, :, None, None]).clamp_(0, 1)

        blur_mask = ((torch.rand(count, device=device) < 0.20) & ~clean)[
            :, None, None, None
        ]
        images = torch.where(blur_mask, self._blur(images), images)

        degradation_draw = torch.rand(count, device=device)
        for upper, resolution_scale in ((0.17, 0.50), (0.34, 0.65), (0.50, 0.80)):
            lower = upper - 0.17
            mask = (
                (degradation_draw >= lower)
                & (degradation_draw < upper)
                & ~clean
            )[:, None, None, None]
            reduced_size = max(1, round(self.image_size * resolution_scale))
            reduced = functional.interpolate(
                images,
                size=(reduced_size, reduced_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            restored = functional.interpolate(
                reduced,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            images = torch.where(mask, restored, images)

        # GPU-friendly codec simulation: quality-dependent color quantization.
        compression_mask = ((torch.rand(count, device=device) < 0.70) & ~clean)[
            :, None, None, None
        ]
        levels = self._uniform(count, 40, 160, device).round()[:, None, None, None]
        quantized = (images * levels).round().div_(levels)
        images = torch.where(compression_mask, quantized, images)

        mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
        return images.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def __call__(
        self, images: list[torch.Tensor], device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full/global and exact/shifted local view pairs."""
        device = torch.device(device)
        batch, heights, widths = self._padded_batch(images, device)
        full_boxes = self._full_boxes(heights, widths)
        global_boxes = self._global_boxes(heights, widths)
        exact_boxes, shifted_boxes = self._tile_boxes(heights, widths)
        return tuple(
            self._appearance(self._crop_resize(batch, boxes))
            for boxes in (full_boxes, global_boxes, exact_boxes, shifted_boxes)
        )
