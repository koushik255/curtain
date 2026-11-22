"""Scene search: which films have this scene, when the exact frame isn't ours.

curtain_ml/scene_modal.py::dense embeds every 8th frame of every film with
pretrained DINOv2 (small or base) into <dir>/vectors.npy (fp16, one row a
frame) and <dir>/frames.json ([film, frame file] a row, grouped by film). A
query is the whole upload squashed to 224 px; a film's score is its best
frame's (the mean of a film's best few did no better).

benchmarks/scene_variants.py compares indexes. On 2026-11-22 the dense base
index, after the SIFT-confirmed films, put the right film #1 for 116 of 180
web-search misses (69 with the first ~1,000-frame small index) and 431 of 475
online-v1 queries (414); the base model costs ~0.35 s a search on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# DINOv2 size by vector width (curtain_ml/scene_modal.py BACKBONES)
BACKBONES = {384: "vit_small_patch14_dinov2.lvd142m", 768: "vit_base_patch14_dinov2.lvd142m",
             1024: "vit_large_patch14_dinov2.lvd142m"}
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SceneSearch:
    def __init__(self, index_dir: Path, model: str = "pretrained"):
        """model: "pretrained" or a scene_modal checkpoint (scene.pt)."""
        import timm

        self.vectors = np.load(Path(index_dir) / "vectors.npy").astype(np.float32)
        backbone = BACKBONES[self.vectors.shape[1]]
        self.model = timm.create_model(backbone, pretrained=model == "pretrained", num_classes=0, img_size=SIZE)
        if model != "pretrained":
            self.model.load_state_dict(torch.load(model, map_location="cpu", weights_only=False)["model"])
        self.model.eval()
        frames = json.loads((Path(index_dir) / "frames.json").read_text())
        self.films = sorted({film for film, _ in frames})
        film_of = np.searchsorted(np.array(self.films), np.array([film for film, _ in frames]))
        if np.any(np.diff(film_of) < 0):
            raise RuntimeError(f"{index_dir}/frames.json is not grouped by film")
        self.spans = np.searchsorted(film_of, np.arange(len(self.films) + 1))
        self.frame_files = [name for _, name in frames]

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
        small = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0)
        x = (np.asarray(small, dtype=np.float32) / 255 - MEAN) / STD
        v = self.model(torch.from_numpy(x).permute(2, 0, 1)[None]).numpy()[0]
        return v / np.linalg.norm(v)

    def search(self, image: Image.Image, top: int = 20) -> list[tuple[str, float, str]]:
        """[(film, score, its best frame's file)] best first."""
        sims = self.vectors @ self.embed(image)
        best = np.maximum.reduceat(sims, self.spans[:-1])
        out = []
        for f in np.argsort(-best)[:top]:
            a, b = self.spans[f], self.spans[f + 1]
            out.append((self.films[f], float(best[f]), self.frame_files[a + int(sims[a:b].argmax())]))
        return out
