"""Reproducible local robustness check against both full 50k-model indexes."""
import hashlib
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from training.train import CurtainEncoder, reference_transform
from training.scoring import normalize_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / 'trained_models/curtain-resnet18-50k-best.pt'
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = CurtainEncoder(checkpoint['config']['embedding_dim']).eval()
    model.load_state_dict(checkpoint['model'])
    transform = reference_transform(checkpoint['config']['image_size'])
    collections = [('l4-50k', root / 'screenshots'),
                   ('l4-50k-lbfive', Path('/home/koushik/lbfive/screenshots'))]
    entries, queries = [], []
    total = 0
    rng = np.random.default_rng(20260908)
    for collection, source in collections:
        directory = root / 'trained_indexes' / collection
        summary = json.loads((directory / 'collection.json').read_text())
        assert summary['checkpoint_sha256'] == digest
        for movie in summary['movies']:
            folder = directory / movie['movie']
            records = json.loads((folder / 'records.json').read_text())
            entries.append((folder, source, movie['movie'], total, len(records)))
            for index in rng.choice(len(records), size=min(2, len(records)), replace=False):
                queries.append((source / movie['movie'] / records[index]['filename'],
                                total + int(index), len(entries) - 1, records[index]))
            total += len(records)
    matrix = np.empty((total, 128), dtype=np.float32)
    for folder, source, name, offset, count in entries:
        stored = np.load(folder / 'embeddings.npy')
        matrix[offset:offset+count] = normalize_rows(stored) if args.normalize else stored
    assert np.isfinite(matrix).all()
    ends = np.array([entry[3] + entry[4] for entry in entries])
    variants = ['original', 'brightness_70pct', 'jpeg_quality_40', 'resize_320px', 'combined']
    results = []
    with torch.inference_mode():
        model(torch.zeros(1, 3, checkpoint['config']['image_size'], checkpoint['config']['image_size']))
        for qi, (path, target, movie_id, record) in enumerate(queries):
            with Image.open(path) as opened:
                original = ImageOps.exif_transpose(opened).convert('RGB')
            for variant in variants:
                image = original
                if variant in ('brightness_70pct', 'combined'):
                    image = ImageEnhance.Brightness(image).enhance(0.7)
                if variant in ('resize_320px', 'combined'):
                    image = image.resize((320, max(1, round(image.height * 320 / image.width))), Image.Resampling.BILINEAR)
                if variant in ('jpeg_quality_40', 'combined'):
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=40)
                    buffer.seek(0)
                    with Image.open(buffer) as jpeg:
                        image = jpeg.convert('RGB')
                start = time.perf_counter()
                query = model(transform(image).unsqueeze(0))[0].numpy()
                if args.normalize:
                    query = normalize_rows(query[None])[0]
                embedded = time.perf_counter()
                scores = matrix @ query
                best = np.argpartition(scores, -5)[-5:]
                best = best[np.argsort(scores[best])[::-1]]
                searched = time.perf_counter()
                predicted = int(best[0])
                predicted_movie = int(np.searchsorted(ends, predicted, side='right'))
                target_score = float(scores[target])
                results.append(dict(query=str(path), variant=variant, target=target,
                    predicted=predicted, predicted_movie=entries[predicted_movie][2],
                    predicted_record_index=predicted-entries[predicted_movie][3],
                    exact_top1=predicted == target, exact_top5=bool(target in best),
                    same_movie=entries[predicted_movie][2].lower() == entries[movie_id][2].lower(),
                    within_2_index_frames=predicted_movie == movie_id and abs(predicted-target) <= 2,
                    target_score=target_score, best_score=float(scores[predicted]),
                    embedding_ms=(embedded-start)*1000, search_ms=(searched-embedded)*1000))
            if (qi+1) % 20 == 0:
                print(f'{qi+1}/{len(queries)} source frames tested', flush=True)
    report = dict(checkpoint_sha256=digest, normalized=args.normalize, seed=20260908, indexed_frames=total,
                  movie_directories=len(entries), source_frames=len(queries), queries=len(results),
                  note='Diagnostic sampled retrieval, not a held-out training benchmark. Strict index-row matches; duplicate or adjacent frames may count as errors. CPU timings exclude disk decode, augmentation, network and startup.',
                  variants={}, results=results)
    for variant in variants:
        rows = [r for r in results if r['variant'] == variant]
        report['variants'][variant] = {key: sum(r[key] for r in rows)/len(rows)
            for key in ('exact_top1', 'exact_top5', 'same_movie', 'within_2_index_frames')}
    report['timings_ms'] = {key: {'median': float(np.median([r[key] for r in results])),
                                'p95': float(np.percentile([r[key] for r in results], 95))}
                            for key in ('embedding_ms', 'search_ms')}
    name = 'collection-robustness-20260908' + ('-normalized' if args.normalize else '')
    output = root / f'trained_models/{name}.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2), flush=True)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
