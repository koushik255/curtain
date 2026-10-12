"""Export a Curtain checkpoint for rcmindex and check the shared-trunk tiling.

Writes:
  trunk.onnx  fp16 ResNet-18 trunk (conv1 .. layer4), NCHW in, [B,512,h/32,w/32] out
  head.bin    fp32 fc weight [dim, 512] followed by fp32 fc bias [dim]
  head.json   embedding size, image size, checkpoint hash

`--check` compares, on CPU, the current per-tile embeddings (crop, resize to
192, full model) with rcmindex's embeddings (resize the frame to 4*192, run the
trunk once, average each 6x6 feature window, apply fc, normalize).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.io import ImageReadMode, decode_jpeg

from curtain_ml.model import CurtainEncoder
from curtain_ml.training import IMAGENET_MEAN, IMAGENET_STD

GRID = 4  # tiles per side of a frame
STEPS = 2 * GRID - 1  # half-stride window positions per side


def trunk_of(model: CurtainEncoder) -> nn.Sequential:
    # Everything before global average pooling.
    b = model.backbone
    return nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3, b.layer4)


def normalized(images: torch.Tensor, size: int) -> torch.Tensor:
    # uint8 NCHW -> resized, ImageNet-normalized float32 (matches the 75k indexer).
    images = functional.interpolate(
        images.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images / 255 - mean) / std


def per_tile(model: CurtainEncoder, frame: torch.Tensor) -> torch.Tensor:
    # The current approach: 49 crops, each run through the full model.
    _, height, width = frame.shape
    th, tw = height // GRID, width // GRID
    tops = [round(i * th / 2) for i in range(STEPS)]
    lefts = [round(i * tw / 2) for i in range(STEPS)]
    tiles = [normalized(frame[None, :, t:t + th, l:l + tw], model.image_size) for t in tops for l in lefts]
    return model(torch.cat(tiles))


def shared_trunk(model: CurtainEncoder, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # rcmindex's approach: one trunk pass at GRID*192, then pooled windows.
    features = trunk_of(model)(normalized(frame[None], GRID * model.image_size))[0]
    cells = features.shape[-1] // GRID  # feature cells per tile (6)
    stride = cells // 2
    pooled = torch.stack([
        features[:, y * stride:y * stride + cells, x * stride:x * stride + cells].mean((1, 2))
        for y in range(STEPS) for x in range(STEPS)
    ])
    tiles = functional.normalize(model.backbone.fc(pooled), dim=1)
    whole = model(normalized(frame[None], model.image_size))[0]
    return tiles, whole


def check(model: CurtainEncoder, frames: list[Path]) -> None:
    cosines = []
    with torch.inference_mode():
        for path in frames:
            frame = decode_jpeg(torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8), mode=ImageReadMode.RGB)
            reference = per_tile(model, frame)
            tiles, _ = shared_trunk(model, frame)
            cosines.append((reference * tiles).sum(1))
            # Can a shared-trunk tile still find its own per-tile twin among the frame's 49?
    cos = torch.cat(cosines).numpy()
    print(f"{len(frames)} frames, {cos.size} tiles")
    print(f"cosine(per-tile, shared-trunk): mean {cos.mean():.4f}  p5 {np.percentile(cos, 5):.4f}  min {cos.min():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, nargs="*", default=[], help="JPEG frames for the CPU comparison")
    args = parser.parse_args()

    model = CurtainEncoder.from_checkpoint(args.checkpoint)
    if args.check:
        check(model, args.check)
    if args.output is None:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    trunk = trunk_of(model).half()
    example = torch.zeros(1, 3, GRID * model.image_size, GRID * model.image_size, dtype=torch.float16)
    torch.onnx.export(
        trunk, (example,), args.output / "trunk.onnx", input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}}, opset_version=17, dynamo=False,
    )
    fc = model.backbone.fc
    with open(args.output / "head.bin", "wb") as head:
        head.write(fc.weight.detach().float().numpy().tobytes())
        head.write(fc.bias.detach().float().numpy().tobytes())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    (args.output / "head.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "embedding_dim": fc.out_features, "features": fc.in_features,
        "image_size": model.image_size, "grid": GRID,
    }, indent=2) + "\n")
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
