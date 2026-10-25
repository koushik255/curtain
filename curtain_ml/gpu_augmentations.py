"""Batched tensor augmentations for CUDA and ROCm training devices."""

from __future__ import annotations

import numpy
import torch
import torch.nn.functional as functional

from curtain_ml.training import IMAGENET_MEAN, IMAGENET_STD


SUBTITLES = (
    "I'm not going anywhere.", "What are you doing here?", "We have to go. Now.",
    "- Where is he?\n- I don't know.", "[DOOR CLOSES]", "You don't understand.",
    "It was never about the money.", "Tell me what happened.", "[THUNDER RUMBLING]",
    "Don't look at me like that.", "I'll be right back.", "He said he'd call.",
    "That's not what I meant.", "Get in the car!", "No one has to know.",
    "- Somewhere over the rainbow -", "Is that clear?", "Why would I lie?",
    "Merci beaucoup.", "Leave it. Let's go.", "Where were you last night?",
    "Keep your voice down.", "We've been here before.", "It's late. Go home.",
)


def _text_bank(size: int, count: int = 96, seed: int = 7) -> torch.Tensor:
    """[count, 4, size, size] RGBA overlays: subtitle lines (white or yellow,
    outlined) near the bottom, and faint corner logos/watermarks."""
    import random

    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    bank = []
    for index in range(count):
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        text = rng.choice(SUBTITLES)
        if index % 4 == 3:  # a logo or watermark in a corner
            font = ImageFont.load_default(size=rng.randint(size // 16, size // 9))
            word = rng.choice(("HBO", "NETFLIX", "BBC", "4K", "CC", "TCM", "@clips", "MUBI"))
            box = draw.textbbox((0, 0), word, font=font)
            x = rng.choice((4, size - (box[2] - box[0]) - 4))
            y = rng.choice((4, size - (box[3] - box[1]) - 6))
            draw.text((x, y), word, font=font, fill=(255, 255, 255, rng.randint(90, 200)))
        else:
            font = ImageFont.load_default(size=rng.randint(size // 18, size // 10))
            box = draw.multiline_textbbox((0, 0), text, font=font, align="center")
            width, height = box[2] - box[0], box[3] - box[1]
            x = max(0, (size - width) // 2 + rng.randint(-size // 20, size // 20))
            y = size - height - rng.randint(size // 20, size // 6)
            colour = (255, 255, 255, 255) if rng.random() < 0.8 else (255, 230, 60, 255)
            draw.multiline_text((x, y), text, font=font, fill=colour, align="center",
                                stroke_width=max(1, size // 96), stroke_fill=(0, 0, 0, 255))
        bank.append(torch.from_numpy(numpy.array(canvas)).permute(2, 0, 1))
    return torch.stack(bank).float().div_(255)


class GpuCombinedAugmenter:
    """Create global and local positive pairs entirely on an accelerator.

    hard=True (the all-films recipe) adds, on top of the original views:
    local positives of other sizes and shapes (0.6-1.4x a tile, aspect 0.7-1.4,
    for tight crops and slivers), subtitles/logos/UI bars on the query-side
    views (global crop and shifted tile), and film grain and sharpening.
    tight=True (with hard) makes those reshaped positives three quarters of
    them and smaller and thinner: 0.4-1.4x a tile, aspect 0.5-2. A tight crop
    (20-35% of the frame) or its 60% corner pieces is 0.5-1.4 of an index tile;
    with 0.6-1.4 tight crops found the exact frame 22 of 40 times.
    paired=True: the batch is (frame, neighbour) pairs; both get the same tile.
    """

    def __init__(self, image_size: int = 192, grid_size: int = 4, hard: bool = False,
                 tight: bool = False):
        if image_size < 1:
            raise ValueError("Image size must be positive")
        if grid_size < 2:
            raise ValueError("Grid size must be at least two")
        self.image_size = image_size
        self.grid_size = grid_size
        self.hard = hard
        self.tight = tight
        self.bank = _text_bank(image_size) if hard else None

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
        self, heights: torch.Tensor, widths: torch.Tensor, paired: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact grid cells and same-sized crops shifted by half a tile
        (with hard, half of them also rescaled and reshaped)."""
        device = heights.device
        count = len(heights)
        if paired:  # a frame and its neighbour: the same cell of each
            tile = torch.randint(self.grid_size * self.grid_size, (count // 2,), device=device)
            tile = tile.repeat_interleave(2)
        else:
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
        crop_widths, crop_heights = tile_widths, tile_heights
        if self.hard:
            # A tight crop or a sliver's strip is rarely exactly a tile: half
            # the positives are 0.6-1.4x a tile, with aspect 0.7-1.4.
            share, smallest, (thinnest, widest) = (0.75, 0.4, (0.5, 2.0)) if self.tight else (0.5, 0.6, (0.7, 1.4))
            reshape = torch.rand(count, device=device) < share
            scale = torch.where(reshape, self._uniform(count, smallest, 1.4, device), 1.0)
            # Aspect drawn in log space, so tall and wide strips are as likely.
            aspect = self._uniform(count, numpy.log(thinnest), numpy.log(widest), device).exp()
            aspect = torch.where(reshape, aspect.sqrt(), 1.0)
            crop_widths = (tile_widths * scale * aspect).minimum(widths)
            crop_heights = (tile_heights * scale / aspect).minimum(heights)
        # Same centre as the cell, then shifted.
        centre_x = left + tile_widths / 2 + delta_x
        centre_y = top + tile_heights / 2 + delta_y
        shifted_left = (centre_x - crop_widths / 2).clamp(min=0)
        shifted_top = (centre_y - crop_heights / 2).clamp(min=0)
        shifted_left = torch.minimum(shifted_left, widths - crop_widths)
        shifted_top = torch.minimum(shifted_top, heights - crop_heights)
        shifted = torch.stack(
            (
                shifted_left,
                shifted_top,
                shifted_left + crop_widths,
                shifted_top + crop_heights,
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

    def _overlay(self, images: torch.Tensor) -> torch.Tensor:
        """Subtitles or a logo (35%), or a phone/player UI bar (10%), on [0, 1] views."""
        device = images.device
        count = len(images)
        draw = torch.rand(count, device=device)
        text = draw < 0.35
        if text.any():
            bank = self.bank.to(device=device, dtype=images.dtype)
            pick = bank[torch.randint(len(bank), (int(text.sum()),), device=device)]
            alpha = pick[:, 3:4]
            images[text] = images[text] * (1 - alpha) + pick[:, :3] * alpha
        bar = (draw >= 0.35) & (draw < 0.45)
        if bar.any():
            rows = torch.arange(self.image_size, device=device)[None, :]
            height = (self._uniform(count, 0.03, 0.10, device) * self.image_size)[:, None]
            top = torch.rand(count, device=device)[:, None] < 0.5
            covered = torch.where(top, rows < height, rows >= self.image_size - height) & bar[:, None]
            shade = torch.where(torch.rand(count, device=device) < 0.7, 0.0, 0.95).to(images.dtype)
            images = torch.where(covered[:, None, :, None], shade[:, None, None, None], images)
        return images

    def _texture(self, images: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Film grain (30%) or sharpening (15%), not on clean views."""
        device = images.device
        count = len(images)
        draw = torch.rand(count, device=device)
        grain = (draw < 0.30) & ~clean
        if grain.any():
            # Grain is a few pixels across: noise at half size, upsampled;
            # strongest in the mid-tones.
            half = self.image_size // 2
            noise = torch.randn(count, 1, half, half, device=device, dtype=images.dtype)
            noise = functional.interpolate(noise, size=(self.image_size,) * 2, mode="bilinear")
            amount = self._uniform(count, 0.01, 0.05, device).to(images.dtype)[:, None, None, None]
            weight = 4 * images.mean(dim=1, keepdim=True) * (1 - images.mean(dim=1, keepdim=True))
            images = torch.where(grain[:, None, None, None], (images + noise * amount * (0.4 + weight)).clamp(0, 1), images)
        sharpen = (draw >= 0.30) & (draw < 0.45) & ~clean
        if sharpen.any():
            amount = self._uniform(count, 0.3, 1.0, device).to(images.dtype)[:, None, None, None]
            sharp = (images + (images - self._blur(images)) * amount).clamp(0, 1)
            images = torch.where(sharpen[:, None, None, None], sharp, images)
        return images

    def _appearance(self, images: torch.Tensor, overlay: bool = False) -> torch.Tensor:
        """Apply lighting, blur, degradation, quantization, and normalization
        (with hard: overlays on query-side views first, and texture)."""
        device = images.device
        count = len(images)
        images = images.div(255).clamp_(0, 1)
        if self.hard and overlay:
            images = self._overlay(images)
        regime = torch.rand(count, device=device)
        clean = regime < 0.20
        mild = (regime >= 0.20) & (regime < 0.70)
        strength = torch.where(mild, 0.35, torch.ones_like(regime))
        strength = torch.where(clean, torch.zeros_like(strength), strength)

        def factor(low: float, high: float) -> torch.Tensor:
            sampled = self._uniform(count, low, high, device)
            return 1 + (sampled - 1) * strength

        def channel_factor(low: float, high: float) -> torch.Tensor:
            sampled = torch.empty(count, 3, device=device).uniform_(low, high)
            return 1 + (sampled - 1) * strength[:, None]

        brightness = factor(0.60, 1.40)
        contrast = factor(0.75, 1.25)
        saturation = factor(0.50, 1.60)
        gamma = factor(0.75, 1.35)
        temperature = self._uniform(count, -0.08, 0.08, device) * strength
        # Another release or transfer of a film is regraded per channel: its own
        # gain and gamma for each channel (tints in any direction, not just
        # warm-cool), crushed or lifted blacks, and S-shaped or flattened tone
        # curves. A Godfather screenshot with 18% less green, 10% less blue and
        # crushed shadows scored 0.57 against its frame, 0.88 once colour-matched.
        channel_gain = channel_factor(0.85, 1.15)
        channel_gamma = channel_factor(0.88, 1.14)  # on top of gamma: kept narrow
        black_point = self._uniform(count, -0.05, 0.06, device) * strength  # > 0 crushes
        curve = self._uniform(count, -0.5, 0.5, device) * strength  # > 0 adds contrast
        greyscale = (torch.rand(count, device=device) < 0.05) & ~clean

        def luma(values: torch.Tensor) -> torch.Tensor:
            return (
                values[:, 0:1] * 0.299
                + values[:, 1:2] * 0.587
                + values[:, 2:3] * 0.114
            )

        images = images * brightness[:, None, None, None]
        contrast_mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - contrast_mean) * contrast[:, None, None, None] + contrast_mean
        gray = luma(images)
        images = gray + (images - gray) * saturation[:, None, None, None]
        exponent = gamma[:, None] * channel_gamma
        images = images.clamp_(1e-4, 1).pow_(exponent[:, :, None, None])
        channel_gains = torch.stack(
            (1 + temperature, torch.ones_like(temperature), 1 - temperature), dim=1
        ) * channel_gain
        images = (images * channel_gains[:, :, None, None]).clamp_(0, 1)
        black = black_point[:, None, None, None]
        images = ((images - black) / (1 - black)).clamp_(0, 1)
        # Monotonic for |curve| <= 0.5: slope 1 + curve * (6x - 6x^2 - 1) > 0.
        bend = curve[:, None, None, None]
        images = images + bend * images * (1 - images) * (2 * images - 1)
        images = torch.where(
            greyscale[:, None, None, None], luma(images).expand_as(images), images
        )

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
        if self.hard:
            images = self._texture(images, clean)

        mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
        return images.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)

    @staticmethod
    def same_pairs(batch: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """[pairs] bool: a (frame, neighbour) pair that is the same picture (a
        static shot: no 1/48-frame block changes by more than 6 levels of 255),
        which no model can tell apart, so is not used as a negative."""
        h, w = int(heights.min()), int(widths.min())
        small = functional.adaptive_avg_pool2d(batch[:, :, :h, :w].float().mean(dim=1, keepdim=True), (27, 48))
        return (small[0::2] - small[1::2]).abs().amax(dim=(1, 2, 3)) <= 6

    @torch.no_grad()
    def __call__(
        self, images: list[torch.Tensor], device: torch.device | str, paired: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Return full/global and exact/shifted local view pairs; with paired,
        also same_pairs for the batch's (frame, neighbour) pairs."""
        device = torch.device(device)
        batch, heights, widths = self._padded_batch(images, device)
        full_boxes = self._full_boxes(heights, widths)
        global_boxes = self._global_boxes(heights, widths)
        exact_boxes, shifted_boxes = self._tile_boxes(heights, widths, paired)
        views = tuple(
            self._appearance(self._crop_resize(batch, boxes), overlay)
            for boxes, overlay in ((full_boxes, False), (global_boxes, True),
                                   (exact_boxes, False), (shifted_boxes, True))
        )
        return (*views, self.same_pairs(batch, heights, widths)) if paired else views
