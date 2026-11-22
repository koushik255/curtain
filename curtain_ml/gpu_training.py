"""Combined global/local contrastive training for CUDA-compatible ROCm builds."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from curtain_ml.gpu_augmentations import GpuCombinedAugmenter, PaddedFrames, pad_frames
from curtain_ml.model import CurtainEncoder
from curtain_ml.training import (
    GridRetrievalDataset,
    RetrievalDataset,
    contrastive_loss,
    discover_movies,
    evaluate,
    seed_everything,
    select_paths,
    split_movies,
)


@dataclass(frozen=True)
class GpuTrainingConfig:
    data_dir: str
    output_dir: str
    epochs: int = 30
    batch_size: int = 256
    image_size: int = 192
    embedding_dim: int = 128
    learning_rate: float = 6e-4
    temperature: float = 0.07
    weight_decay: float = 1e-4
    workers: int = 20
    eval_movies: int = 6
    seed: int = 42
    limit: int | None = None
    view_mode: str = "combined_gpu"
    grid_size: int = 4
    tile_jitter: float = 0.5
    global_loss_weight: float = 0.5
    local_loss_weight: float = 0.5
    forward_chunk_size: int = 2048
    evaluation_interval: int = 2
    init: str | None = None  # checkpoint to fine-tune from instead of random weights
    freeze_bn: bool = False  # keep BatchNorm running statistics from init
    # Loader batches per optimizer step. The contrastive loss then sees
    # batch_size * accumulate sources (its negatives) while host memory holds
    # one loader batch and the GPU one forward chunk (gradient caching).
    accumulate: int = 1
    progress_interval: int = 10  # optimizer steps between progress lines
    # Hard negatives from the same shot: each source comes with a frame up to
    # this many frames later in its film (0.5-1.5 s at 2 fps), in the same
    # batch, so the loss must tell neighbouring frames apart. 0: off.
    neighbours: int = 0
    hard: bool = False  # GpuCombinedAugmenter(hard=True): reshaped tiles, overlays, grain
    tight: bool = False  # with hard: smaller, thinner reshaped tiles, more of them
    # A codec_pool folder (<film>/<frame>.jpg): with this chance a source's
    # query-side views are cropped from its codec-damaged copy, where it has one.
    degraded: str | None = None
    degrade_chance: float = 0.6
    # Pairs with a degraded copy (e.g. a real web screenshot warped onto its
    # frame, curtain_ml/web_pairs.py) are trained on this many times an epoch.
    degraded_repeat: int = 1
    web_colour: bool = False  # GpuCombinedAugmenter(web_colour=True): web regrades on query views
    # Speed only (a run can resume with these changed, see SPEED_FIELDS).
    # bf16 needs A100/L40S/H100-class GPUs and no loss scaler; fp16 runs anywhere.
    precision: str = "fp16"
    compile: bool = False  # torch.compile the encoder and the augmenter's per-pixel chain
    # Gradient caching re-runs every forward chunk (about a quarter more model
    # compute) so only one chunk's activations are held at a time. "auto": try
    # the whole step at once and switch to caching for the run if it runs out
    # of memory; "on"/"off" force it.
    gradient_caching: str = "auto"
    # > 0: profile this many optimizer steps (after 3 warm-up steps), write
    # profile.txt and profile.json to output_dir, and stop.
    profile_steps: int = 0

    def __post_init__(self):
        if self.global_loss_weight < 0 or self.local_loss_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.global_loss_weight + self.local_loss_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if self.forward_chunk_size < 1:
            raise ValueError("Forward chunk size must be positive")
        if self.evaluation_interval < 1:
            raise ValueError("Evaluation interval must be positive")
        if self.accumulate < 1:
            raise ValueError("Accumulate must be positive")
        if self.precision not in ("fp16", "bf16"):
            raise ValueError("Precision must be fp16 or bf16")
        if self.gradient_caching not in ("auto", "on", "off"):
            raise ValueError("Gradient caching must be auto, on or off")


# Settings that change speed, not what is learned: a run resumes with them changed.
SPEED_FIELDS = {"workers", "forward_chunk_size", "progress_interval", "precision", "compile",
                "gradient_caching", "profile_steps"}


def decode(path: Path) -> torch.Tensor:
    # libwebp/libjpeg straight into a uint8 [3, H, W] tensor: the same pixels
    # as PIL's decode, without PIL's image object and the copy out of it.
    return decode_image(str(path), mode=ImageReadMode.RGB, apply_exif_orientation=True)


class DecodedFrameDataset(Dataset):
    """Decode JPEGs on CPU; every learned transformation happens on the GPU.
    With degraded, each item is (frame, query copy): the frame's codec_pool
    copy with probability chance (where it has one), else the frame itself."""

    def __init__(self, paths: list[Path], degraded: Path | None = None, chance: float = 0.0):
        self.paths = paths
        self.degraded = degraded
        self.chance = chance

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.item(self.paths[index])

    def item(self, path: Path) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image = decode(path)
        if self.degraded is None:
            return image
        copy = self.degraded / path.parent.name / f"{path.stem}.jpg"
        if random.random() < self.chance and copy.is_file():
            damaged = decode(copy)
            if damaged.shape == image.shape:
                return image, damaged
        return image, image


class NeighbourPairDataset(Dataset):
    """(frame, neighbour) pairs, decoded like DecodedFrameDataset."""

    def __init__(self, pairs: list[tuple[Path, Path]], degraded: Path | None = None, chance: float = 0.0):
        self.pairs = pairs
        self.frames = DecodedFrameDataset([], degraded, chance)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> list:
        return [self.frames.item(path) for path in self.pairs[index]]


def flatten_pairs(items: list[list]) -> list:
    return [image for pair in items for image in pair]


def split_copies(items: list) -> list | tuple[list[torch.Tensor], list[torch.Tensor]]:
    # (frame, query copy) items -> (frames, query copies); plain frames unchanged.
    if items and isinstance(items[0], tuple):
        return [frame for frame, _ in items], [copy for _, copy in items]
    return items


def split_pairs(items: list[list]) -> list | tuple[list[torch.Tensor], list[torch.Tensor]]:
    return split_copies(flatten_pairs(items))


def collate_padded(items: list) -> tuple[PaddedFrames, PaddedFrames | None]:
    """Collate in the worker into (frames, query copies or None), each padded
    into one tensor, so the loader pins it and the GPU gets one copy."""
    images = split_pairs(items) if items and isinstance(items[0], list) else split_copies(items)
    images, queries = images if isinstance(images, tuple) else (images, None)
    return pad_frames(images), pad_frames(queries) if queries is not None else None


def frame_number(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else -1


def neighbour_pairs(movies: dict[str, list[Path]], names: list[str], reach: int,
                    limit: int | None, seed: int) -> list[tuple[Path, Path]]:
    """Disjoint (frame, later frame within reach) pairs from each film, by
    frame number (a folder may hold only a sample, its pairs included)."""
    rng = random.Random(seed)
    pairs = []
    for name in names:
        frames = sorted(movies[name], key=frame_number)
        numbers = [frame_number(path) for path in frames]
        used = set()
        for i, number in enumerate(numbers):
            if i in used:
                continue
            later = [j for j in range(i + 1, min(len(frames), i + reach + 1))
                     if j not in used and 0 < numbers[j] - number <= reach]
            if later:
                j = rng.choice(later)
                used |= {i, j}
                pairs.append((frames[i], frames[j]))
    if limit is not None and limit < len(pairs):
        pairs = rng.sample(pairs, limit)
    return pairs


def _loaders(
    config: GpuTrainingConfig,
    train_paths: list[Path] | list[tuple[Path, Path]],
    eval_paths: list[Path],
) -> tuple[DataLoader, dict[str, DataLoader]]:
    # Variable-sized decoded images are returned as a list. Filesystem-backed
    # sharing avoids exhausting one file descriptor per tensor at large batch
    # sizes while workers pass those tensors to the training process.
    multiprocessing.set_sharing_strategy("file_system")
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    degraded = Path(config.degraded).expanduser() if config.degraded else None
    train_loader = DataLoader(
        NeighbourPairDataset(train_paths, degraded, config.degrade_chance) if config.neighbours
        else DecodedFrameDataset(train_paths, degraded, config.degrade_chance),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_padded,
        # Full-resolution decoded frames are large. One queued batch per worker
        # keeps host/shared-memory use bounded on large logical GPU batches.
        prefetch_factor=1 if config.workers else None,
        **common,
    )
    evaluation_common = {
        **common,
        "batch_size": min(config.batch_size, 512),
    }
    return train_loader, {
        "global": DataLoader(
            RetrievalDataset(eval_paths, config.image_size),
            shuffle=False,
            drop_last=False,
            **evaluation_common,
        ),
        "grid": DataLoader(
            GridRetrievalDataset(
                eval_paths,
                config.image_size,
                config.seed + 100_000,
                config.grid_size,
                config.tile_jitter,
            ),
            shuffle=False,
            drop_last=False,
            **evaluation_common,
        ),
    }


def save_atomic(state: dict, path: Path) -> None:
    # A crash mid-write leaves the old file, never a truncated one.
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    partial.replace(path)


def write_profile(profiler, output_dir: Path, steps: int) -> dict[str, float]:
    """profile.txt (ops by GPU time, then by CPU time) and profile.json (a
    trace for chrome://tracing or ui.perfetto.dev) in output_dir."""
    averages = profiler.key_averages()
    table = "\n\n".join(
        averages.table(sort_by=key, row_limit=40) for key in ("cuda_time_total", "cpu_time_total"))
    (output_dir / "profile.txt").write_text(table + "\n")
    profiler.export_chrome_trace(str(output_dir / "profile.json"))
    print(averages.table(sort_by="cuda_time_total", row_limit=25), flush=True)
    return {"profiled_steps": steps}


def train(config: GpuTrainingConfig,
          on_checkpoint: Callable[[], None] | None = None) -> dict[str, float]:
    """Train one encoder on global and 4x4-local objectives.

    Every epoch ends with output_dir/last.pt (model, optimizer, scheduler,
    scaler, best score) and a line in metrics.jsonl, then on_checkpoint (on
    Modal: volume.commit). A run started again on the same output_dir with
    the same config resumes after the last finished epoch."""
    seed_everything(config.seed)
    data_dir = Path(config.data_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    movies = discover_movies(data_dir)
    train_movies, eval_movies = split_movies(movies, config.eval_movies, config.seed)
    if config.neighbours:  # batch_size and limit count pairs: 2 sources each
        train_paths = neighbour_pairs(movies, train_movies, config.neighbours, config.limit, config.seed)
    else:
        train_paths = select_paths(movies, train_movies, config.limit, config.seed)
    if config.degraded and config.degraded_repeat > 1:
        copies = Path(config.degraded).expanduser()

        def has_copy(item) -> bool:
            return any((copies / p.parent.name / f"{p.stem}.jpg").is_file()
                       for p in (item if isinstance(item, tuple) else (item,)))
        repeated = [item for item in train_paths if has_copy(item)]
        train_paths = train_paths + repeated * (config.degraded_repeat - 1)
        print(f"{len(repeated)} sources with a degraded copy, each {config.degraded_repeat}x", flush=True)
    eval_paths = select_paths(movies, eval_movies, config.limit, config.seed + 1)
    if len(train_paths) < config.batch_size or len(eval_paths) < 2:
        raise RuntimeError("The selected split is too small for this configuration.")

    split_manifest = {
        "training_movies": train_movies,
        "evaluation_movies": eval_movies,
        "training_frames": len(train_paths) * (2 if config.neighbours else 1),
        "evaluation_frames": len(eval_paths),
    }
    last_path = output_dir / "last.pt"
    history_path = output_dir / "metrics.jsonl"
    resumed = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
    if resumed is not None:
        # Fields the checkpoint predates are taken to match.
        changed = sorted(k for k, v in asdict(config).items()
                         if k not in SPEED_FIELDS and k in resumed["config"] and resumed["config"][k] != v)
        if changed:
            raise RuntimeError(f"{last_path} is from a run with a different config ({', '.join(changed)}); "
                               "use another output_dir or delete it to start over.")
    (output_dir / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Combined training requires a CUDA or ROCm PyTorch build.")

    device = torch.device("cuda")
    # Fixed shapes all run long: let cuDNN time its algorithms once and keep
    # the fastest. TF32 for the float32 matmuls left (the loss's logits).
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_loader, evaluation_loaders = _loaders(config, train_paths, eval_paths)
    augmenter = GpuCombinedAugmenter(config.image_size, config.grid_size, config.hard, config.tight,
                                     config.web_colour)
    model = CurtainEncoder(config.embedding_dim, config.image_size)
    if config.init:
        model.load_state_dict(torch.load(config.init, map_location="cpu", weights_only=False)["model"])
    model = model.to(device)
    model.to(memory_format=torch.channels_last)
    try:  # one kernel for every parameter's update instead of several per tensor
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=True
        )
    except (RuntimeError, ValueError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    warmup_epochs = max(1, min(3, config.epochs // 5))

    def learning_rate_multiplier(scheduler_epoch: int) -> float:
        completed_epoch = scheduler_epoch + 1
        if completed_epoch <= warmup_epochs:
            return completed_epoch / warmup_epochs
        progress = (completed_epoch - warmup_epochs) / max(
            1, config.epochs - warmup_epochs
        )
        return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    autocast_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    # bf16 has float32's range: nothing to scale, and no inf check (a sync) per step.
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    # forward shares model's parameters; model stays uncompiled for evaluation
    # (other batch sizes) and so its state_dict keys are unchanged.
    forward = torch.compile(model) if config.compile else model
    if config.compile:
        augmenter.compile()
    cache_gradients = {"on": True, "off": False}.get(config.gradient_caching)  # None: first step decides
    best_score = -1.0
    final_metrics: dict[str, float] = {}
    start_epoch = 1
    history = history_path.read_text().splitlines() if history_path.exists() else []
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        if scaler.is_enabled() and resumed["scaler"]:  # empty from a bf16 run
            scaler.load_state_dict(resumed["scaler"])
        best_score = resumed["best_score"]
        final_metrics = resumed["metrics"]
        start_epoch = resumed["epoch"] + 1
        print(f"resuming {output_dir} after epoch {resumed['epoch']} (best score {best_score:.4f})", flush=True)
        del resumed
    # Lines past the checkpoint's epoch (a crash after logging them) go.
    history = [line for line in history if json.loads(line)["epoch"] < start_epoch]
    history_path.write_text("".join(f"{line}\n" for line in history))
    print(
        json.dumps(
            {
                **split_manifest,
                "device": str(device),
                "rocm": torch.version.hip,
                "gpu": torch.cuda.get_device_name(0),
            },
            indent=2,
        ),
        flush=True,
    )

    def train_step(views: torch.Tensor, same_pairs: torch.Tensor | None, batch_sources: int,
                   cached: bool) -> torch.Tensor:
        """One optimizer step; returns [loss, global, local] on the device."""
        optimizer.zero_grad(set_to_none=True)
        chunks = range(0, len(views), config.forward_chunk_size)
        with torch.profiler.record_function("model"), torch.autocast("cuda", dtype=autocast_dtype):
            # Gradient caching: embed every chunk without keeping its
            # activations, take the loss over all of them, then re-run each
            # chunk with gradients and feed it its share of the loss gradient.
            with torch.set_grad_enabled(not cached):
                embeddings = torch.cat(
                    [forward(views[offset : offset + config.forward_chunk_size]) for offset in chunks]
                )
            if cached:
                embeddings = embeddings.detach().requires_grad_()
            (
                full_embedding,
                global_embedding,
                exact_embedding,
                shifted_embedding,
            ) = embeddings.reshape(batch_sources, 4, -1).unbind(dim=1)
            global_loss = contrastive_loss(
                full_embedding, global_embedding, config.temperature, same_pairs
            )
            local_loss = contrastive_loss(
                exact_embedding, shifted_embedding, config.temperature, same_pairs
            )
            loss = (
                config.global_loss_weight * global_loss
                + config.local_loss_weight * local_loss
            )
        with torch.profiler.record_function("backward"):
            scaler.scale(loss).backward()
            if cached:
                for offset in chunks:
                    end = offset + config.forward_chunk_size
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        chunk = forward(views[offset:end])
                    chunk.backward(embeddings.grad[offset:end])
        with torch.profiler.record_function("optimizer"):
            scaler.step(optimizer)
            scaler.update()
        return torch.stack((loss, global_loss, local_loss)).detach().float()

    profiler = None
    profile_started = 0
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        if config.freeze_bn:
            # Small fine-tuning batches re-estimate the statistics badly and
            # shift every layer; keep init's (the affine weights still train).
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        started = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats()
        totals = torch.zeros(3, device=device)  # loss, global, local: read only at progress lines
        steps = sources = 0
        data_wait = 0.0
        # Each loader batch's GPU work between two events. They fire in stream
        # order, so their spans add up to the time the GPU was busy; the rest of
        # the epoch it waited for the CPU (decoding, mostly).
        spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        pending = []
        batches = iter(train_loader)
        while True:
            waited = time.perf_counter()
            try:
                images = next(batches)
            except StopIteration:
                break
            data_wait += time.perf_counter() - waited
            if profiler is None and config.profile_steps and steps == 3:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                profiler.start()
                profile_started = steps
            span = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            span[0].record()
            # Augment each loader batch on the GPU at once (small 192 px views)
            # and step once `accumulate` of them are collected.
            images, queries = images
            with torch.profiler.record_function("augment"):
                pending.append(augmenter(images, device, paired=bool(config.neighbours), queries=queries))
            if len(pending) < config.accumulate:
                span[1].record()
                spans.append(span)
                continue
            # Interleave view families (source 0's four views, then source
            # 1's, ...) so BatchNorm sees the same distribution in every forward
            # chunk. Capping the physical forward batch also avoids MIOpen's
            # very-large-N BatchNorm kernel limit while the contrastive loss
            # still uses the complete logical source batch. Each view is copied
            # once, straight into a channels-last [sources * 4, 3, H, W] tensor.
            batch_sources = sum(len(augmented[0]) for augmented in pending)
            size = config.image_size
            interleaved = torch.empty((batch_sources, 4, size, size, 3), device=device,
                                      dtype=pending[0][0].dtype)
            offset = 0
            for augmented in pending:
                count = len(augmented[0])
                for family in range(4):  # channels-last views: permuted, they are [n, H, W, 3] in memory
                    interleaved[offset : offset + count, family].copy_(augmented[family].permute(0, 2, 3, 1))
                offset += count
            views = interleaved.view(batch_sources * 4, size, size, 3).permute(0, 3, 1, 2)
            same_pairs = torch.cat([augmented[4] for augmented in pending]) if config.neighbours else None
            pending = []
            if cache_gradients is None:
                try:
                    losses = train_step(views, same_pairs, batch_sources, cached=False)
                    cache_gradients = False
                except torch.OutOfMemoryError:
                    losses = None
                if losses is None:  # outside the except: its traceback no longer holds the tensors
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    cache_gradients = True
                    losses = train_step(views, same_pairs, batch_sources, cached=True)
                print(f"  gradient caching {'on' if cache_gradients else 'off'} "
                      f"({len(views)} views a step)", flush=True)
            else:
                losses = train_step(views, same_pairs, batch_sources, cached=cache_gradients)
            span[1].record()
            spans.append(span)
            totals += losses
            steps += 1
            sources += batch_sources
            if profiler is not None and steps - profile_started == config.profile_steps:
                profiler.stop()
                return write_profile(profiler, output_dir, config.profile_steps)
            if steps % config.progress_interval == 0:
                elapsed = time.perf_counter() - started
                print(f"  epoch {epoch}: step {steps}, {sources} sources, "
                      f"{sources / elapsed:.0f}/s, loss {totals[0].item() / steps:.4f}", flush=True)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        gpu_busy = sum(begin.elapsed_time(end) for begin, end in spans) / 1000
        scheduler.step()

        steps = max(1, steps)
        total_loss, total_global, total_local = totals.tolist()
        training_metrics = {
            "epoch": epoch,
            "training_loss": total_loss / steps,
            "global_loss": total_global / steps,
            "local_loss": total_local / steps,
            "learning_rate": epoch_learning_rate,
            "training_seconds": training_seconds,
            # Near 1: the GPU is the limit. Well below: it waits on data
            # loading (data_wait_seconds is the CPU blocked on the loader).
            "gpu_busy_fraction": gpu_busy / training_seconds,
            "data_wait_seconds": data_wait,
            "source_frames_per_second": sources / training_seconds,
            "model_views_per_second": sources * 4 / training_seconds,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        should_evaluate = (
            epoch == 1
            or epoch % config.evaluation_interval == 0
            or epoch == config.epochs
        )
        if should_evaluate:
            global_metrics = evaluate(model, evaluation_loaders["global"], device)
            grid_metrics = evaluate(model, evaluation_loaders["grid"], device)
            selection_score = 0.5 * (
                global_metrics["exact_frame_top1"]
                + grid_metrics["exact_frame_top1"]
            )
            final_metrics = {
                **training_metrics,
                "global_exact_frame_top1": global_metrics["exact_frame_top1"],
                "global_movie_top1": global_metrics["movie_top1"],
                "grid_exact_frame_top1": grid_metrics["exact_frame_top1"],
                "grid_movie_top1": grid_metrics["movie_top1"],
                "selection_score": selection_score,
                "evaluated": True,
                "epoch_seconds": time.perf_counter() - started,
            }
            checkpoint = {
                "model": model.state_dict(),
                "config": asdict(config),
                "metrics": final_metrics,
            }
            if selection_score > best_score:
                best_score = selection_score
                save_atomic(checkpoint, output_dir / "best.pt")
        else:
            final_metrics = {
                **training_metrics,
                "evaluated": False,
                "epoch_seconds": time.perf_counter() - started,
            }
        print(json.dumps(final_metrics), flush=True)
        with history_path.open("a") as history_file:
            history_file.write(json.dumps(final_metrics) + "\n")
        save_atomic({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "best_score": best_score,
            "metrics": final_metrics,
        }, last_path)
        if on_checkpoint is not None:
            on_checkpoint()

    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n"
    )
    return final_metrics


def parse_args() -> GpuTrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/data/training/selected_screenshots"))
    parser.add_argument("--output", type=Path, default=Path("training_runs/combined"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--forward-chunk-size", type=int, default=2048)
    parser.add_argument("--evaluation-interval", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--init", type=Path, help="checkpoint to fine-tune from")
    parser.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm statistics from --init")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="loader batches per step: batch-size * accumulate negatives")
    parser.add_argument("--neighbours", type=int, default=0,
                        help="pair each frame with one up to this many frames later (hard negatives)")
    parser.add_argument("--hard", action="store_true", help="reshaped tiles, overlays, grain")
    parser.add_argument("--tight", action="store_true", help="with --hard: smaller, thinner tiles")
    parser.add_argument("--eval-movies", type=int, default=6)
    parser.add_argument("--degraded", type=Path, help="codec_pool folder: query views from codec-damaged copies")
    parser.add_argument("--degrade-chance", type=float, default=0.6)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--compile", action="store_true", help="torch.compile the encoder and augmenter")
    parser.add_argument("--gradient-caching", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="profile this many steps into the output folder, then stop")
    args = parser.parse_args()
    return GpuTrainingConfig(
        data_dir=str(args.data),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        limit=args.limit,
        forward_chunk_size=args.forward_chunk_size,
        evaluation_interval=args.evaluation_interval,
        learning_rate=args.learning_rate,
        init=str(args.init) if args.init else None,
        freeze_bn=args.freeze_bn,
        accumulate=args.accumulate,
        neighbours=args.neighbours,
        hard=args.hard,
        tight=args.tight,
        eval_movies=args.eval_movies,
        degraded=str(args.degraded) if args.degraded else None,
        degrade_chance=args.degrade_chance,
        precision=args.precision,
        compile=args.compile,
        gradient_caching=args.gradient_caching,
        profile_steps=args.profile_steps,
    )


if __name__ == "__main__":
    train(parse_args())
