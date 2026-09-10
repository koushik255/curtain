"""Diagnostic query-only crop sweep; does not modify the live search service."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from training.server import TrainedSearchEngine, DEFAULT_CHECKPOINT, DEFAULT_INDEX, DEFAULT_FRAMES


def main():
    torch.set_num_threads(4)
    engine = TrainedSearchEngine(DEFAULT_CHECKPOINT, DEFAULT_INDEX, DEFAULT_FRAMES, 'cpu')
    path = Path('/home/koushik/.codex/attachments/fbe31928-114a-4458-8b96-910e74189ec2/codex-clipboard-0c24ed5d-c193-44aa-938c-d1257244e533.png')
    with Image.open(path) as opened:
        source = ImageOps.exif_transpose(opened).convert('RGB')
    width, height = source.size
    views = [('full', (0, 0, width, height))]
    for scale in (0.9, 0.8, 0.65):
        w, h = round(width*scale), round(height*scale)
        for position, x, y in [('center',(width-w)//2,(height-h)//2), ('top_left',0,0),
                               ('top_right',width-w,0), ('bottom_left',0,height-h),
                               ('bottom_right',width-w,height-h)]:
            views.append((f'{scale}_{position}', (x,y,x+w,y+h)))
    for ratio in (2.2, 2.39, 1.0):
        w, h = min(width,round(height*ratio)), min(height,round(width/ratio))
        for position, fraction in [('start',0), ('center',0.5), ('end',1)]:
            x,y=round((width-w)*fraction),round((height-h)*fraction)
            views.append((f'aspect_{ratio}_{position}',(x,y,x+w,y+h)))
    target_indices=np.array([i for i,r in enumerate(engine.records) if r['movie']=='Lawrence_of_Arabia_1962'])
    report=[]
    max_scores=np.full(len(engine.records),-np.inf,dtype=np.float32)
    mean_scores=np.zeros(len(engine.records),dtype=np.float32)
    def describe(index, score):
        r=engine.records[int(index)]
        return dict(movie=r['movie'], filename=r['filename'], score=float(score), path=str(engine.frame_path(int(index))))
    with torch.inference_mode():
        for name, box in views:
            query=engine.model(engine.transform(source.crop(box)).unsqueeze(0))[0].numpy()
            query/=np.linalg.norm(query)
            scores=engine.embeddings@query
            max_scores=np.maximum(max_scores,scores)
            mean_scores+=scores/len(views)
            best=int(np.argmax(scores))
            target=int(target_indices[np.argmax(scores[target_indices])])
            row=dict(view=name,box=box,top1=describe(best,scores[best]),
                     best_lawrence=describe(target,scores[target]),
                     lawrence_rank=int(np.sum(scores>scores[target]))+1)
            report.append(row)
            print(json.dumps(row),flush=True)
    combined={}
    for name,scores in [('maximum_across_views',max_scores),('mean_across_views',mean_scores)]:
        best=int(np.argmax(scores))
        target=int(target_indices[np.argmax(scores[target_indices])])
        combined[name]=dict(top1=describe(best,scores[best]),best_lawrence=describe(target,scores[target]),
                            lawrence_rank=int(np.sum(scores>scores[target]))+1)
    output=Path('trained_models/multicrop-lawrence-experiment.json')
    output.write_text(json.dumps(dict(query=str(path),reference_frames=len(engine.records),
        views=report,combined=combined, caveat='Single-image query-only experiment; crops of indexed frames were not tested.'),indent=2)+'\n')
    print('COMBINED',json.dumps(combined),flush=True)


if __name__=='__main__':
    main()
