"""Paired lighting robustness comparison with model-specific reference vectors."""
import hashlib
import io
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from training.train import CurtainEncoder, reference_transform

CONDITIONS = ('original', 'brightness_0.7', 'brightness_0.6', 'brightness_1.4',
              'gamma_0.7', 'gamma_1.4', 'contrast_0.7', 'warm', 'cool',
              'jpeg_resize', 'combined')
CROP_CONDITIONS = ('original', 'crop_center_0.9', 'crop_center_0.8', 'crop_topright_0.8',
                   'crop_center_0.65', 'crop_width_0.75', 'crop_combined', 'combined')


def alter(image, condition):
    if condition.startswith('crop_'):
        if condition not in CROP_CONDITIONS:
            raise ValueError(condition)
        w,h = image.size
        scale = 0.8 if condition == 'crop_combined' else float(condition.rsplit('_',1)[-1])
        cw,ch = round(w*scale), (h if condition == 'crop_width_0.75' else round(h*scale))
        x,y = ((w-cw,0) if condition == 'crop_topright_0.8' else ((w-cw)//2,(h-ch)//2))
        cropped = image.crop((x,y,x+cw,y+ch))
        return alter(cropped,'combined') if condition == 'crop_combined' else cropped
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition.startswith('brightness_'):
        return ImageEnhance.Brightness(image).enhance(float(condition.split('_')[1]))
    if condition.startswith('gamma_'):
        return TF.adjust_gamma(image, float(condition.split('_')[1]))
    if condition == 'contrast_0.7':
        return ImageEnhance.Contrast(image).enhance(0.7)
    if condition in ('warm', 'cool'):
        shift = 0.1 if condition == 'warm' else -0.1
        return Image.merge('RGB', tuple(channel.point([min(255, round(i*gain)) for i in range(256)])
            for channel, gain in zip(image.split(), (1+shift, 1., 1-shift))))
    if condition == 'combined':
        image = ImageEnhance.Brightness(image).enhance(0.7)
        image = TF.adjust_gamma(image, 1.2)
    if condition in ('combined', 'jpeg_resize'):
        image = image.resize((320, max(1, round(image.height*320/image.width))), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=40)
        buffer.seek(0)
        with Image.open(buffer) as opened:
            return opened.convert('RGB')
    return image.copy()


class Images(Dataset):
    def __init__(self, paths, size, condition):
        self.paths, self.transform, self.condition = paths, reference_transform(size), condition

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
        return self.transform(alter(image, self.condition))


@torch.inference_mode()
def encode(model, paths, size, condition, device):
    batches = []
    for images in DataLoader(Images(paths, size, condition), batch_size=256,
                             num_workers=8, pin_memory=True):
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(images.to(device, non_blocking=True))
        batches.append(torch.nn.functional.normalize(output.float(), dim=1))
    return torch.cat(batches)


@torch.inference_mode()
def run(data, checkpoints, splits, output, conditions=CONDITIONS, external_queries=None):
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    first_split = splits['original']
    assert all(first_split == split for split in splits.values()), 'Training splits differ'
    paths = sorted(p for p in data.glob('*/*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    movies = sorted({p.parent.name for p in paths})
    movie_ids = torch.tensor([movies.index(p.parent.name) for p in paths], device=device)
    rng = random.Random(20260909)
    targets = []
    for movie in first_split['evaluation_movies']:
        candidates = [i for i,p in enumerate(paths) if p.parent.name == movie]
        targets.extend(sorted(rng.sample(candidates, min(128, len(candidates)))))
    query_paths = [paths[i] for i in targets]
    target = torch.tensor(targets, device=device)
    result = dict(seed=20260909, reference_frames=len(paths), query_frames=len(targets),
                  conditions=list(conditions), evaluation_movies=first_split['evaluation_movies'],
                  queries=[str(p.relative_to(data)) for p in query_paths], models={},
                  caveat='Validation movies used for checkpoint selection; not untouched test movies. Strict frame IDs; duplicates may count as misses. Reference pool size is recorded; not full production index.')
    for name, checkpoint_path in checkpoints.items():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = saved['config']
        model = CurtainEncoder(cfg['embedding_dim']).to(device).eval()
        model.load_state_dict(saved['model'])
        print(f'Encoding {len(paths)} reference frames for {name}', flush=True)
        references = encode(model, paths, cfg['image_size'], 'original', device)
        metrics = {}
        for condition in conditions:
            queries = encode(model, query_paths, cfg['image_size'], condition, device)
            scores = queries @ references.T
            best = scores.topk(5, dim=1).indices
            top1 = best[:, 0] == target
            top5 = (best == target[:, None]).any(dim=1)
            movie = movie_ids[best[:, 0]] == movie_ids[target]
            metrics[condition] = dict(top1=float(top1.float().mean()), top5=float(top5.float().mean()),
                movie_top1=float(movie.float().mean()), correct=top1.cpu().tolist(),
                predicted=[str(paths[i].relative_to(data)) for i in best[:, 0].cpu().tolist()])
            print(json.dumps(dict(model=name, condition=condition,
                                  **{k:v for k,v in metrics[condition].items() if k not in ('correct', 'predicted')})), flush=True)
            del scores, queries
        result['models'][name] = dict(checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                                     config=cfg, metrics=metrics)
        external = {}
        for label, payload in (external_queries or {}).items():
            with Image.open(io.BytesIO(payload)) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
            with torch.autocast('cuda', dtype=torch.float16):
                query = model(reference_transform(cfg['image_size'])(image).unsqueeze(0).to(device))
            query = torch.nn.functional.normalize(query.float(),dim=1)
            scores = (query @ references.T)[0]
            best = scores.topk(5).indices.tolist()
            external[label] = [dict(frame=str(paths[i].relative_to(data)),score=float(scores[i])) for i in best]
            print('EXTERNAL',name,label,json.dumps(external[label]),flush=True)
        result['models'][name]['external_queries'] = external
        del model, references
    result['elapsed_seconds'] = time.perf_counter()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result
