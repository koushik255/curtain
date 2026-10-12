"""Shared DigitalOcean Spaces settings for upload_frames.py and launch.py.

Each movie is stored as ~100 MB parts plus a manifest written last:
frames/<collection>/<movie>/part-NNNN.tar (members <movie>/*.jpg) and
frames/<collection>/<movie>/manifest.json. The key lives in ~/.config/curtain/spaces.env and
never leaves this machine; droplets get presigned download URLs instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REGION = "tor1"  # same region as the MI325X droplets
BUCKET = "curtain-frames-koushik"
PREFIX = "frames"
KEY_FILE = Path.home() / ".config/curtain/spaces.env"
ROOT = Path(__file__).resolve().parents[2]
# Local frame folders: <root>/<movie>/*.jpg, stored as <collection>/<movie>.
SOURCES = {
    "full-curtain": ROOT / ".local/screenshots",
    "l4-50k-lbfive": Path.home() / "lbfive/screenshots",
}


def ensure_boto3() -> None:
    # The system python has no boto3; rerun the same command through uv.
    try:
        import boto3  # noqa: F401
    except ImportError:
        if os.environ.get("CURTAIN_UV_REEXEC"):
            sys.exit("boto3 unavailable even under uv")
        os.environ["CURTAIN_UV_REEXEC"] = "1"
        os.execvp("uv", ["uv", "run", "--quiet", "--no-project", "--with", "boto3",
                         "python", sys.argv[0], *sys.argv[1:]])


def client():
    import boto3

    values = dict(line.split("=", 1) for line in KEY_FILE.read_text().split())
    return boto3.client(
        "s3", region_name=REGION, endpoint_url=f"https://{REGION}.digitaloceanspaces.com",
        aws_access_key_id=values["SPACES_KEY"], aws_secret_access_key=values["SPACES_SECRET"],
    )


def local_movies() -> list[tuple[str, Path]]:
    return [(collection, d) for collection, root in SOURCES.items() if root.is_dir()
            for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("*.jpg"))]


def all_keys(s3) -> dict[str, int]:
    """Every object under the prefix -> size."""
    keys = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX + "/"):
        for item in page.get("Contents", []):
            keys[item["Key"]] = item["Size"]
    return keys


def stored_movies(s3, keys: dict[str, int] | None = None) -> set[tuple[str, str]]:
    """(collection, movie) for every fully uploaded movie (manifest present)."""
    keys = all_keys(s3) if keys is None else keys
    return {(k.split("/")[1], k.split("/")[2]) for k in keys if k.endswith("/manifest.json")}


def part_links(s3, collection: str, movie: str, hours: int = 12) -> list[str]:
    """Presigned GET URLs for one stored movie's parts, in order."""
    import json

    base = f"{PREFIX}/{collection}/{movie}"
    manifest = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{base}/manifest.json")["Body"].read())
    return [s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": f"{base}/{part}"},
                                      ExpiresIn=hours * 3600) for part in manifest["parts"]]
