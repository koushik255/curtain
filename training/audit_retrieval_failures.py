"""Inspect failed queries without altering source screenshots or the benchmark."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'trained_models/collection-robustness-20260908-normalized.json').read_text())
    entries = []
    offset = 0
    for name, source in [('l4-50k', root / 'screenshots'),
                         ('l4-50k-lbfive', Path('/home/koushik/lbfive/screenshots'))]:
        index = root / 'trained_indexes' / name
        for movie in json.loads((index / 'collection.json').read_text())['movies']:
            records = json.loads((index / movie['movie'] / 'records.json').read_text())
            entries.append((offset, offset+len(records), source / movie['movie'], records))
            offset += len(records)
    failures = []
    for row in report['results']:
        if row['exact_top1']:
            continue
        prediction = row['predicted']
        start, end, folder, records = next(e for e in entries if e[0] <= prediction < e[1])
        predicted = folder / records[prediction-start]['filename']
        with Image.open(row['query']) as opened:
            query = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
        with Image.open(predicted) as opened:
            candidate = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
        identical = query.shape == candidate.shape and np.array_equal(query, candidate)
        failures.append(dict(variant=row['variant'], query=row['query'], prediction=str(predicted),
                             identical_source_pixels=identical, source_pixel_std=float(query.std()),
                             source_pixel_mean=float(query.mean()), same_movie=row['same_movie'],
                             within_2_index_frames=row['within_2_index_frames'],
                             score_gap=row['best_score']-row['target_score']))
    output = root / 'trained_models/retrieval-failure-audit-20260908.json'
    output.write_text(json.dumps(failures, indent=2)+'\n')
    for variant in report['variants']:
        rows = [r for r in failures if r['variant'] == variant]
        print(variant, dict(failures=len(rows), identical_source_pixels=sum(r['identical_source_pixels'] for r in rows),
                            low_variation_sources=sum(r['source_pixel_std'] < 2 for r in rows),
                            nearby_frames=sum(r['within_2_index_frames'] for r in rows)))
    print('Original failures:', json.dumps([r for r in failures if r['variant']=='original'], indent=2))


if __name__ == '__main__':
    main()
