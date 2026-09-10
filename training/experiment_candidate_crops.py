"""Reference-crop reranking diagnostic; production index and service unchanged."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from training.server import TrainedSearchEngine, DEFAULT_CHECKPOINT, DEFAULT_INDEX, DEFAULT_FRAMES


def crops(image):
    w,h=image.size
    yield 'full', image
    for scale in (.9,.8,.65):
        cw,ch=round(w*scale),round(h*scale)
        for label,x,y in [('center',(w-cw)//2,(h-ch)//2),('tl',0,0),('tr',w-cw,0),('bl',0,h-ch),('br',w-cw,h-ch)]:
            yield f'{scale}_{label}',image.crop((x,y,x+cw,y+ch))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    engine=TrainedSearchEngine(DEFAULT_CHECKPOINT,DEFAULT_INDEX,DEFAULT_FRAMES,'cpu')
    previous=json.loads(Path('trained_models/multicrop-lawrence-experiment.json').read_text())
    with Image.open(previous['query']) as opened:
        query_image=ImageOps.exif_transpose(opened).convert('RGB')
    query_tensors=torch.stack([engine.transform(query_image.crop(v['box'])) for v in previous['views']])
    queries=engine.model(query_tensors).numpy()
    queries/=np.linalg.norm(queries,axis=1,keepdims=True)
    maximum=np.full(len(engine.records),-np.inf,dtype=np.float32)
    full_scores=None
    reserved=set()
    for n,query in enumerate(queries):
        scores=engine.embeddings@query
        if n==0: full_scores=scores.copy()
        count=20 if n==0 else 3
        reserved.update(np.argpartition(scores,-count)[-count:].tolist())
        np.maximum(maximum,scores,out=maximum)
    pool=np.argpartition(maximum,-100)[-100:]
    for index in pool[np.argsort(maximum[pool])[::-1]]:
        if len(reserved)>=100: break
        reserved.add(int(index))
    candidates=sorted(reserved,key=lambda i:float(maximum[i]),reverse=True)
    known=[i for i,r in enumerate(engine.records) if r['movie']=='Lawrence_of_Arabia_1962' and r['filename']=='frame_003599.jpg'][0]
    nearby=[i for i in candidates if engine.records[i]['movie']=='Lawrence_of_Arabia_1962' and 3570 <= engine.records[i]['frame_number'] <= 3650]
    print('Shortlist:',len(candidates),'known wide shot present:',known in candidates,'nearby wide-shot candidates:',len(nearby),flush=True)
    rows=[]
    # An oracle probe is reported separately; never included in retrieval rankings.
    for n,index in enumerate(candidates + ([] if known in candidates else [known])):
        path=engine.frame_path(index)
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
        views=list(crops(image))
        embeddings=engine.model(torch.stack([engine.transform(im) for _,im in views])).numpy()
        embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True)
        scores=queries@embeddings.T
        a,b=np.unravel_index(np.argmax(scores),scores.shape)
        r=engine.records[index]
        row=dict(index=index,movie=r['movie'],filename=r['filename'],path=str(path),
                 shortlisted=index in candidates,full_score=float(full_scores[index]),
                 full_query_to_reference_crops=float(scores[0].max()),
                 any_query_to_reference_crops=float(scores.max()),
                 blended=float(.5*full_scores[index]+.5*scores[0].max()),
                 best_query_view=previous['views'][a]['view'],best_reference_view=views[b][0])
        rows.append(row)
        if (n+1)%20==0: print('Compared',n+1,'candidates',flush=True)
    rankings={key:sorted([r for r in rows if r['shortlisted']],key=lambda r:r[key],reverse=True)[:10]
              for key in ('full_query_to_reference_crops','any_query_to_reference_crops','blended')}
    result=dict(query=previous['query'],reference_frames=len(engine.records),candidate_count=len(candidates),
                shortlist_method='20 full-image candidates plus 3 per query crop; fill remaining slots by maximum score',
                known_wide_shot_shortlisted=known in candidates,nearby_scene_candidates=nearby,
                oracle_probe=next(r for r in rows if r['index']==known),rankings=rankings,rows=rows)
    Path('trained_models/candidate-crops-lawrence-experiment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:items[:3] for key,items in rankings.items()},indent=2),flush=True)
    print('ORACLE',json.dumps(result['oracle_probe']),flush=True)


if __name__=='__main__': main()
