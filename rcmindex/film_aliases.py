"""Catalogue names that are another film: duplicates and mislabelled footage.

film_aliases.json maps a movie folder name to {"is": the film it really is,
"why": ...}. The site lists both folders as that one film. Found by comparing
films' scene vectors (rcmindex/scene_search.py) and checked by eye on
2026-11-22. Mislabelled footage of a film we already have under its own name
goes in excluded_sources.json instead (e.g. The_Return_2003).
"""

from __future__ import annotations

import json
from pathlib import Path

ALIASES = {name: entry["is"] for name, entry in
           json.loads(Path(__file__).with_name("film_aliases.json").read_text()).items()}


def canonical(name: str) -> str:
    return ALIASES.get(name, name)
