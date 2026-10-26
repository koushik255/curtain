"""Real-world encodings: what a frame goes through before someone pastes it.

Each recipe runs the codecs a frame really meets on its way to the web, with
the settings those places use: ffmpeg's x264, x265, SVT-AV1, VP9, MPEG-2 and
GIF encoders for streams, rips and clips; libjpeg and libwebp (through
Pillow) for the image sites; and the colour bugs of player screenshots.

Video recipes encode a short clip, so the target frame comes out as an
inter-coded frame, as in a real stream. Built from the indexed frames, the
clip is the target's neighbours (0.5 s apart, so motion prediction is weaker
than in the film; quantisation, chroma subsampling, deblocking and grain
loss are real). `clip_from_video` gives the real footage when a file exists.

    from curtain_ml.encodings import RECIPES, encode
    image = encode("stream_h264", frames, index, random.Random(0))
"""

from __future__ import annotations

import io
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"


# ------------------------------------------------------------------ codecs

def transcode(frames: list[Image.Image], output: list[str], ext: str, vf: str = EVEN,
              decode_vf: str = "", fps: float = 24) -> list[Image.Image]:
    """Encode frames into a file with ffmpeg and decode every frame back."""
    size = frames[0].size
    raw = b"".join(f.convert("RGB").resize(size).tobytes() for f in frames)
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / f"clip.{ext}"
        ffmpeg(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                "-i", "-", "-vf", vf, *output, "-y", str(clip)], raw)
        ffmpeg(["-i", str(clip), *(["-vf", decode_vf] if decode_vf else []),
                "-fps_mode", "passthrough", str(Path(tmp) / "%05d.png")])
        return [Image.open(p).convert("RGB") for p in sorted(Path(tmp).glob("*.png"))]


def ffmpeg(args: list[str], stdin: bytes | None = None) -> None:
    # Encoders like SVT-AV1 log to stderr regardless of -v; show it only on failure.
    done = subprocess.run(["ffmpeg", "-v", "error", *args], input=stdin, capture_output=True)
    if done.returncode:
        raise RuntimeError(f"ffmpeg {' '.join(args)}\n{done.stderr.decode(errors='replace')[-2000:]}")


def still(image: Image.Image, format: str, **options) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=format, **options)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def fit(image: Image.Image, long_side: int) -> Image.Image:
    # Downscale only, as upload sites do.
    scale = long_side / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def height(rng: random.Random, choices: list[int]) -> str:
    return f"scale=-2:{rng.choice(choices)}:flags=bicubic"


# ------------------------------------------------------------------ recipes
# Each takes the clip, the target's position in it and a seeded rng, and
# returns the target frame as it comes out.

Recipe = Callable[[list[Image.Image], int, random.Random], Image.Image]


def stream_h264(frames, index, rng):
    # Low and middle tiers of a streaming service or a pirate stream site.
    return transcode(frames, ["-c:v", "libx264", "-preset", "fast", "-tune", "film",
                              "-crf", str(rng.randint(26, 34)), "-pix_fmt", "yuv420p"],
                     "mp4", vf=height(rng, [360, 480, 540]) if rng.random() < 0.5 else EVEN)[index]


def stream_hevc(frames, index, rng):
    return transcode(frames, ["-c:v", "libx265", "-preset", "fast", "-x265-params", "log-level=error",
                              "-crf", str(rng.randint(28, 36)), "-pix_fmt", "yuv420p"],
                     "mp4", vf=height(rng, [480, 540]) if rng.random() < 0.5 else EVEN)[index]


def stream_av1_grain(frames, index, rng):
    # AV1 with grain synthesis, as Netflix ships it: the real grain is
    # denoised away and a synthetic one painted back on.
    return transcode(frames, ["-c:v", "libsvtav1", "-preset", "8", "-crf", str(rng.randint(40, 52)),
                              "-svtav1-params", f"film-grain={rng.randint(6, 14)}:film-grain-denoise=1",
                              "-pix_fmt", "yuv420p"], "mkv")[index]


def youtube_vp9(frames, index, rng):
    # A film clip on YouTube, watched at 360p or 480p.
    return transcode(frames, ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
                              "-crf", str(rng.randint(36, 46)), "-b:v", "0", "-pix_fmt", "yuv420p"],
                     "webm", vf=height(rng, [360, 480]))[index]


def dvd_mpeg2(frames, index, rng):
    # Anamorphic 720x480 MPEG-2 at DVD bitrates, shown at 854x480.
    return transcode(frames, ["-c:v", "mpeg2video", "-b:v", f"{rng.randint(3, 6)}M", "-maxrate", "9M",
                              "-bufsize", "1835k", "-g", "15", "-bf", "2", "-pix_fmt", "yuv420p"],
                     "mpg", vf="scale=720:480:flags=bicubic,setsar=32/27",
                     decode_vf="scale=854:480:flags=bicubic,setsar=1")[index]


def upscaled_sharpened(frames, index, rng):
    # A small copy blown up and sharpened: "4K remaster" uploads and AI upscales.
    low = rng.choice([270, 360])
    return transcode(frames, ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"],
                     "mp4", vf=f"scale=-2:{low}:flags=bilinear,scale=-2:{low * 3}:flags=lanczos,"
                               f"unsharp=5:5:{rng.uniform(0.8, 1.6):.2f}")[index]


def gif(frames, index, rng):
    # A reaction GIF: small, 256 colours, dithered.
    width = rng.choice([320, 400, 480])
    dither = rng.choice(["bayer:bayer_scale=3", "sierra2_4a", "floyd_steinberg", "none"])
    return transcode(frames, [], "gif", vf=f"scale={width}:-1:flags=lanczos,split[a][b];"
                                           f"[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither={dither}")[index]


def tv_range(frames, index, rng):
    # Player screenshot that never expanded limited range: grey blacks, dim whites.
    x = np.asarray(stream_h264(frames, index, rng), dtype=np.float32)
    return Image.fromarray(np.round(16 + x * 219 / 255).astype(np.uint8))


def wrong_matrix(frames, index, rng):
    # HD video decoded with the SD colour matrix (or the reverse): a hue shift.
    pair = rng.choice(["bt709:bt601", "bt601:bt709"])
    return transcode(frames, ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"],
                     "mp4", decode_vf=f"colormatrix={pair}")[index]


def twitter(frames, index, rng):
    image = fit(frames[index], rng.choice([680, 1200, 2048]))
    return still(image, "JPEG", quality=85, subsampling="4:2:0", progressive=True)


def reddit_preview(frames, index, rng):
    image = fit(frames[index], rng.choice([640, 960]))
    return still(image, "WEBP", quality=rng.randint(70, 85))


def whatsapp(frames, index, rng):
    image = fit(frames[index], 1600)
    return still(image, "JPEG", quality=rng.randint(45, 70), subsampling="4:2:0")


def reupload_chain(frames, index, rng):
    # Screenshot of a stream, then passed around: three or four sites in a row.
    image = stream_h264(frames, index, rng)
    for site in rng.sample([twitter, reddit_preview, whatsapp, twitter, whatsapp], rng.randint(3, 4)):
        image = site([image], 0, rng)
    return image


RECIPES: dict[str, Recipe] = {
    "stream_h264": stream_h264,
    "stream_hevc": stream_hevc,
    "stream_av1_grain": stream_av1_grain,
    "youtube_vp9": youtube_vp9,
    "dvd_mpeg2": dvd_mpeg2,
    "upscaled_sharpened": upscaled_sharpened,
    "gif": gif,
    "tv_range": tv_range,
    "wrong_matrix": wrong_matrix,
    "twitter": twitter,
    "reddit_preview": reddit_preview,
    "whatsapp": whatsapp,
    "reupload_chain": reupload_chain,
}


def encode(name: str, frames: list[Image.Image], index: int, rng: random.Random) -> Image.Image:
    return RECIPES[name](frames, index, rng)


def clip_from_video(path: Path, seconds: float, before: float = 1.0, after: float = 1.0
                    ) -> tuple[list[Image.Image], int]:
    """Decode the real footage around a moment; returns the frames and the target's index."""
    start = max(0.0, seconds - before)
    with tempfile.TemporaryDirectory() as tmp:
        ffmpeg(["-ss", f"{start:.3f}", "-i", str(path), "-t", f"{before + after:.3f}",
                "-fps_mode", "passthrough", str(Path(tmp) / "%05d.png")])
        frames = [Image.open(p).convert("RGB") for p in sorted(Path(tmp).glob("*.png"))]
    fps = len(frames) / (before + after)
    return frames, min(len(frames) - 1, round((seconds - start) * fps))
