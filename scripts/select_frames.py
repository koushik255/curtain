from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from curtain_ml.training import IMAGE_SUFFIXES


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[index * (len(paths) - 1) // (count - 1)] for index in range(count)]


def select_frames(source: Path, destination: Path, total: int) -> dict:
    movies = {
        directory.name: sorted(
            path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for directory in sorted(source.iterdir())
        if directory.is_dir()
    }
    movies = {name: paths for name, paths in movies.items() if paths}
    if not movies:
        raise RuntimeError(f"No movie frames found in {source}.")
    if total > sum(len(paths) for paths in movies.values()):
        raise ValueError("Requested more frames than the source contains.")

    base, remainder = divmod(total, len(movies))
    counts = {
        name: base + (position < remainder)
        for position, name in enumerate(sorted(movies))
    }
    if any(counts[name] > len(movies[name]) for name in movies):
        raise ValueError("At least one movie does not contain enough frames for a balanced selection.")

    manifest: dict[str, object] = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "requested_frames": total,
        "movies": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, paths in movies.items():
        output_dir = destination / name
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = evenly_spaced(paths, counts[name])
        for source_path in selected:
            output_path = output_dir / source_path.name
            if output_path.exists():
                continue
            try:
                os.link(source_path, output_path)
            except OSError:
                shutil.copy2(source_path, output_path)
        manifest["movies"][name] = {
            "available": len(paths),
            "selected": len(selected),
            "first": selected[0].name,
            "last": selected[-1].name,
        }

    (destination / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced, evenly spaced movie frames.")
    parser.add_argument("--source", type=Path, default=Path("screenshots"))
    parser.add_argument("--output", type=Path, default=Path("selected_screenshots_50k"))
    parser.add_argument("--total", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = select_frames(args.source, args.output, args.total)
    print(json.dumps({"movies": len(result["movies"]), "frames": result["requested_frames"]}))
