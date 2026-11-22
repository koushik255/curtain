"""Compare retrieval depth on a fixed, failure-enriched development diagnostic.

No query labels enter candidate retrieval or ranking. The test split is never
loaded as input to the experiment. Results are movie-level, not exact-frame truth.
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmarks.online_benchmark import OUT, ROOT, digest, read, save

SETTINGS = {'current':(64,512,40), 'deeper_alignment':(64,512,200),
            'wider_retrieval':(256,2048,200)}


def main():
    config=read(OUT/'run-config.json')
    selection=read(OUT/'candidate-experiment-selection.json')
    assert digest((OUT/'frozen.json').read_bytes())==selection['frozen_sha256']
    os.environ['CURTAIN_FAISS_DIR']=config['index']
    import cv2
    import faiss
    import torch
    from apps import server
    from rcmindex import search
    assert digest(search.EXCLUSIONS_FILE.read_bytes())==selection['exclusions_sha256']
    torch.set_num_threads(4);cv2.setNumThreads(1)
    engine=server.TrainedSearchEngine(Path(config['checkpoint']))
    ivf=faiss.extract_index_ivf(engine.search_index['index'])
    original_search=search.search;original_cap=server.ALIGN_CANDIDATES;original_probe=ivf.nprobe
    queries={q['id']:q for q in read(OUT/'frozen.json')['queries']}
    destination=OUT/'candidate-experiment';destination.mkdir(exist_ok=True)
    metadata={'notes':__doc__,'selection':selection,'settings':SETTINGS,
              'checkpoint_sha256':digest(Path(config['checkpoint']).read_bytes()),
              'implementation_sha256':digest(Path(__file__).read_bytes())}
    save(destination/'config.json',metadata)
    rows=[]
    try:
        for number,query_id in enumerate(selection['ids'],1):
            q=queries[query_id];assert q['partition']=='development'
            path=OUT/'queries'/(query_id+'.jpg')
            with Image.open(path) as source:image=source.convert('RGB')
            readings=engine.embed_query(image)
            row={'id':query_id,'movie':q['movie'],'query':str(path),'settings':{},'exact_truth':None}
            for name,(probe,tiles,cap) in SETTINGS.items():
                ivf.nprobe=probe;server.ALIGN_CANDIDATES=cap
                search.search=partial(original_search,candidates=tiles)
                started=time.perf_counter();ranked,_=engine.rank(image,readings)
                elapsed=(time.perf_counter()-started)*1000
                candidates=[]
                for index,score,alignment in ranked:
                    candidates.append({**engine.records[index],'index':int(index),'score':float(score),
                                       'points':int(alignment[0]) if alignment else 0,
                                       'path':str(engine.frame_path(index))})
                correct=lambda c:c['movie'].casefold()==q['movie'].casefold()
                row['settings'][name]={'top1_movie':bool(candidates) and correct(candidates[0]),
                    'movie_in_20':any(correct(c) for c in candidates),'ms_without_embedding':elapsed,
                    'results':candidates}
            rows.append(row);save(destination/'results.json',{'metadata':metadata,'rows':rows})
            print(number,query_id,{k:v['top1_movie'] for k,v in row['settings'].items()},flush=True)
    finally:
        ivf.nprobe=original_probe;server.ALIGN_CANDIDATES=original_cap;search.search=original_search
    summary={name:{'images':len(rows),'movie_top1':sum(r['settings'][name]['top1_movie'] for r in rows),
                    'movie_in_20':sum(r['settings'][name]['movie_in_20'] for r in rows),
                    'movie_recoveries':sum(not r['settings']['current']['top1_movie'] and r['settings'][name]['top1_movie'] for r in rows),
                    'movie_regressions':sum(r['settings']['current']['top1_movie'] and not r['settings'][name]['top1_movie'] for r in rows)} for name in SETTINGS}
    save(destination/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    for start in range(0,len(rows),4):
        group=rows[start:start+4];canvas=Image.new('RGB',(1600,260*len(group)),'#202020');draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(group):
            panels=[('query',row['query'])]+[(name,data['results'][0]['path']) for name,data in row['settings'].items()]
            for x,(label,path) in enumerate(panels):
                with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(390,210))
                canvas.paste(fitted,(x*400+(400-fitted.width)//2,y*260+30))
                draw.text((x*400+3,y*260+3),row['id'] if x==0 else label,fill='white')
                if x:draw.text((x*400+3,y*260+245),row['settings'][label]['results'][0]['movie'][:55],fill='white')
        canvas.save(destination/f'comparison-{start//4:02d}.jpg',quality=90)


if __name__=='__main__':main()
