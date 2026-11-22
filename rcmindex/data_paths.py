"""Where Curtain's images live. Every frame on this machine is under .local/data/:

  frames/originals/<collection>/<movie>/frame_NNNNNN.jpg   original JPEGs (whole only for some films)
  frames/webp/<collection>/<movie>/frame_NNNNNN.webp       640 px WebP copies (the only copy of most films)
  unindexed_films/<movie>/                                  lbfive films not in the index yet
                                                            (~/lbfive/screenshots is a symlink here)
  training/                                                 selected_screenshots*, the 50k tar
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".local/data"
ORIGINALS = DATA / "frames/originals"
WEBP = DATA / "frames/webp"
UNINDEXED = DATA / "unindexed_films"
TRAINING = DATA / "training"
COLLECTIONS = ("full-curtain", "l4-50k-lbfive")
# Original JPEG folders per index collection.
ORIGINAL_DIRS = {collection: ORIGINALS / collection for collection in COLLECTIONS}
