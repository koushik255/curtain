"""Does the scene model put the right film in the 20 results? (curtain_ml/scene_modal.py)

Inputs: the site's stored results for every query (web-search-v2 originals,
online-v1 live run) and .local/benchmarks/scene/<model>.json (each query's
films by scene score; web-search-v2 has it for the site's misses only, whose
images were kept, and a hit's #1 never changes under these policies).

Policies for the 20 films shown:
    site      the site as it is (films in result order)
    scene     the scene model alone
    mixed     the site's confirmed films (SIFT-verified) first, then the
              site's other films and the scene model's alternately
    scene2nd  confirmed first, then scene films, then the site's others

    .venv/bin/python -m benchmarks.scene_eval pretrained scene-v1
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / ".local/benchmarks"


def films(results: list[dict], confirmed: bool | None = None) -> list[str]:
    out = []
    for r in results:
        if confirmed is not None and bool(r.get("confirmed")) != confirmed:
            continue
        if r["movie"] not in out:
            out.append(r["movie"])
    return out


def merge(policy: str, site: list[dict], scene: list[str] | None) -> list[str]:
    if scene is None or policy == "site":
        return films(site)[:20]
    if policy == "scene":
        return scene[:20]
    shown = films(site, confirmed=True)
    rest = [f for f in films(site) if f not in shown]
    if policy == "mixed":
        queue = [x for pair in zip(rest + [None] * len(scene), scene) for x in pair if x]
    else:
        queue = scene + rest
    for f in queue:
        if f not in shown:
            shown.append(f)
    return shown[:20]


def main() -> None:
    models = sys.argv[1:] or ["pretrained"]
    v2 = [json.loads(l) for l in (B / "web-search-v2/results.jsonl").read_text().splitlines()]
    reviews = json.loads((B / "web-search-v2/reviews.json").read_text())
    v2 = [r for r in v2 if r["variant"] == "original" and reviews.get(r["id"]) != "exclude"]
    v1 = [json.loads(l) for l in (B / "online-v1-live/results.jsonl").read_text().splitlines()]
    v1 = [r for r in v1 if r.get("variant", "original") == "original"]
    cases = [("web-search-v2 " + reviews[r["id"]], "v2/" + r["id"], r) for r in v2] + \
            [("online-v1", "v1/" + r["id"], r) for r in v1]
    print(f"| model | policy | set | n | #1 | top 5 | top 20 |\n|---|---|---|---|---|---|---|")
    for model in models:
        ranked = json.loads((B / "scene" / f"{model}.json").read_text())
        for policy in ("site", "scene", "mixed", "scene2nd"):
            if policy == "site" and model != models[0]:
                continue
            tally = defaultdict(lambda: [0, 0, 0, 0])
            for group, key, r in cases:
                scene = [f for f, *_ in ranked[key]] if key in ranked else None
                if policy == "scene" and scene is None:  # a hit whose image wasn't kept
                    continue
                shown = merge(policy, r["results"], scene)
                for g in (group, "all web-search-v2" if group.startswith("web") else None):
                    if g is None:
                        continue
                    t = tally[g]
                    t[0] += 1
                    t[1] += shown[:1] == [r["movie"]]
                    t[2] += r["movie"] in shown[:5]
                    t[3] += r["movie"] in shown[:20]
            for g in sorted(tally):
                n, a, b, c = tally[g]
                print(f"| {model if policy != 'site' else '-'} | {policy} | {g} | {n} | {a} | {b} | {c} |")


if __name__ == "__main__":
    main()
