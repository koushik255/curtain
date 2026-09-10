"""Movie-level training provenance tied to the active checkpoint."""
import hashlib
import json
from pathlib import Path

LABELS = {'training': 'Training', 'validation': 'Validation',
          'unseen': 'Unseen by training/validation', 'unknown': 'Unknown provenance'}
EXPLANATION = ('Training: some frames were used to learn the model. Validation: not used for '
               'gradient training, but used to select checkpoints. Unseen: absent from both saved '
               'lists; not necessarily untouched by later benchmarks. These are movie-level labels, '
               'not a claim that every screenshot was used in training.')


def movie_key(name):
    return name.casefold()


def load_split(checkpoint, checkpoint_hash, project_root):
    for folder in (checkpoint.parent, project_root / 'trained_models/l4-50k'):
        split, best = folder / 'split.json', folder / 'best.pt'
        if split.exists() and best.exists() and hashlib.sha256(best.read_bytes()).hexdigest() == checkpoint_hash:
            return json.loads(split.read_text())
    return None


def category(name, split):
    if split is None:
        return 'unknown'
    key = movie_key(name)
    if key in {movie_key(n) for n in split['training_movies']}:
        return 'training'
    if key in {movie_key(n) for n in split['evaluation_movies']}:
        return 'validation'
    return 'unseen'
