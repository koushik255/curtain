"""Send the scene benchmark's queries to a running server, as the page uploads them.

The web-search-v2 misses (misses-site/img/*.q.jpg) and every online-v1 query,
each shrunk to 640 px and saved as JPEG 90 (apps/templates/index.html), to
--url's /api/search-stream; one JSON line per query in --out (resumable).
Then the right film's #1 / top-5 / top-20 counts, next to the site's stored
results for the same queries.

    .venv/bin/python -m benchmarks.scene_live --url http://127.0.0.1:8783 --out .local/benchmarks/scene/live-pretrained.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from rcmindex.film_aliases import canonical

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / ".local/benchmarks"


def upload(path: Path) -> bytes:
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
    if max(im.size) > 640:
        im = ImageOps.contain(im, (640, 640), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def search(url: str, data: bytes) -> dict:
    request = urllib.request.Request(url + "/api/search-stream", data=data, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        events = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    return next(e for e in reversed(events) if e.get("type") == "result")


def films(results: list[dict]) -> list[str]:
    out = []
    for r in results:  # rcmindex/film_aliases.json: duplicates count as their film
        if canonical(r["movie"]) not in out:
            out.append(canonical(r["movie"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8783")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reviews = json.loads((B / "web-search-v2/reviews.json").read_text())
    v2 = {json.loads(l)["id"]: json.loads(l) for l in (B / "web-search-v2/results.jsonl").read_text().splitlines()
          if json.loads(l)["variant"] == "original"}
    v1 = {json.loads(l)["id"]: json.loads(l) for l in (B / "online-v1-live/results.jsonl").read_text().splitlines()}
    cases = [("web-search-v2 " + reviews[p.name[:-6]], p, v2[p.name[:-6]])
             for p in sorted((B / "web-search-v2/misses-site/img").glob("*.q.jpg"))
             if reviews.get(p.name[:-6]) != "exclude"]
    cases += [("online-v1", B / "online-v1/queries" / f"{i}.jpg", r) for i, r in sorted(v1.items())]
    done = {}
    if args.out.exists():
        done = {json.loads(l)["key"]: json.loads(l) for l in args.out.read_text().splitlines()}
    with args.out.open("a") as out:
        for n, (group, path, old) in enumerate(cases, 1):
            key = f"{group}/{old['id']}"
            if key in done:
                continue
            result = search(args.url, upload(path))
            row = {"key": key, "group": group, "movie": old["movie"], "films": films(result["results"]),
                   "match": [r.get("match", "frame") for r in result["results"]],
                   "seconds": result["timings"]["server_total_ms"] / 1000}
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[key] = row
            if n % 50 == 0:
                print(f"{n}/{len(cases)}", flush=True)
    tally = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    for group, path, old in cases:
        row = done[f"{group}/{old['id']}"]
        before, movie = films(old["results"]), canonical(old["movie"])
        row = dict(row, films=[canonical(f) for f in row["films"]])
        t = tally[group]
        t[0] += 1
        t[1] += before[:1] == [movie]; t[2] += movie in before[:20]
        t[3] += row["films"][:1] == [movie]; t[4] += movie in row["films"][:5]
        t[5] += movie in row["films"][:20]; t[6] += row["seconds"]
    print("| set | n | #1 before | #1 now | top 5 now | top 20 before | top 20 now | mean s |")
    print("|---|---|---|---|---|---|---|---|")
    for group, (n, b1, b20, a1, a5, a20, s) in sorted(tally.items()):
        print(f"| {group} | {n} | {b1} | {a1} | {a5} | {b20} | {a20} | {s / n:.2f} |")


if __name__ == "__main__":
    main()
