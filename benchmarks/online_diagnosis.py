"""Explain development failures with a labelled-movie oracle, outside accuracy.

The known movie is scanned exhaustively only to diagnose candidate recall and
reference coverage. Oracle frames are never inserted in benchmark results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.online_benchmark import OUT, ROOT, read, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--documented-only',action='store_true')
    args=parser.parse_args()
    config=read(args.out/'run-config.json')
    os.environ['CURTAIN_FAISS_DIR']=config['index']
    import cv2
    import torch
    from apps.server import TrainedSearchEngine
    from rcmindex import search
    from benchmarks.template_rerank import match
    cv2.setNumThreads(1);torch.set_num_threads(args.threads)
    rows=[json.loads(line) for line in (args.out/'results.jsonl').read_text().splitlines()]
    if args.documented_only:
        documented={r['id'] for r in read(args.out/'source-provenance.json')['records'] if r['provenance']=='documented_frame_capture'}
        rows=[r for r in rows if r['id'] in documented]
    # Fixed deterministic subset: first failed query of each development movie.
    cases=[];seen=set()
    for row in sorted(rows,key=lambda r:r['id']):
        if row['partition']!='development' or row['before_movie'] or row['movie'] in seen:continue
        seen.add(row['movie']);cases.append(row)
        if len(cases)==args.limit:break
    engine=TrainedSearchEngine(Path(config['checkpoint']));output=[]
    for row in cases:
        with Image.open(row['query']) as im:image=im.convert('RGB')
        readings=engine.embed_query(image)
        frames,pool_scores=search.candidate_frames(engine.search_index,readings)
        order=np.argsort(-pool_scores,kind='stable')
        movie_ranks=[rank for rank,position in enumerate(order,1) if engine.records[int(frames[position])]['movie'].casefold()==row['movie'].casefold()]
        movie=engine.movie_lookup[row['movie']]
        indices=np.concatenate([np.arange(start,start+count) for start,count in movie['spans']])
        scores=np.empty(len(indices),dtype=np.float32)
        for start in range(0,len(indices),512):
            batch=indices[start:start+512]
            vectors=search.frame_vectors(engine.search_index,batch)
            scores[start:start+len(batch)]=search.scores(vectors,readings).max(axis=0)
        results=[];features=__import__('rcmindex.align',fromlist=['features']).features(np.asarray(image.convert('L')))
        for position in np.argsort(-scores,kind='stable')[:8]:
            index=int(indices[position]);path=engine.frame_path(index)
            points,matrix,size=engine.locate(features,index)
            with Image.open(path) as im:gray=np.asarray(im.convert('L'))
            results.append({**engine.records[index],'index':index,'path':str(path),'score':float(scores[position]),
                            'points':points,'matrix':matrix,'ncc':match(np.asarray(image.convert('L')),gray),
                            'exact_truth':None})
        result={'id':row['id'],'movie':row['movie'],'query':row['query'],'partition':'development',
                'candidate_pool_frames':len(frames),'first_movie_rank_before_sift':min(movie_ranks) if movie_ranks else None,
                'movie_in_returned_20':row['movie_in_candidates'],'film_scanned_frames':len(indices),
                'best_film_embedding_candidates':results,
                'interpretation':'Labelled-movie oracle; separate from unassisted retrieval, not exact-frame truth'}
        filename='documented-diagnosis.json' if args.documented_only else 'development-diagnosis.json'
        output.append(result);save(args.out/filename,{'selection':'First failure per development movie, sorted ID, capped at limit','documented_only':args.documented_only,'limit':args.limit,'rows':output})
        print(row['id'],'first movie pool rank',result['first_movie_rank_before_sift'],
              'oracle points',max(r['points'] for r in results),flush=True)


if __name__=='__main__':main()
