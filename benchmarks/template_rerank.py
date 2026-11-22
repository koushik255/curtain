"""Experimental template matching of returned candidates; never inject labels.

This is an offline diagnostic, not a production reranker or a match verifier.
It can establish whether pixel evidence survives when sparse SIFT features do
not, and whether the missing exact frame is already available to a reranker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rcmindex.data_paths import ORIGINAL_DIRS, WEBP

ROOT = Path(__file__).resolve().parents[1]


def frame_path(result: dict) -> Path:
    source = result['source']
    small = WEBP / source / (Path(result['filename']).stem+'.webp')
    if small.exists():
        return small
    collection,movie=source.split('/')
    directory = ORIGINAL_DIRS[collection]
    return directory/movie/result['filename']


def gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def match(query: np.ndarray, reference: np.ndarray) -> float:
    # Normalized correlation removes an affine brightness/contrast change.
    # Scale search is in fractions of reference width, independent of labels.
    qh,qw=query.shape;rh,rw=reference.shape
    best=-1.
    for fraction in (0.10,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.85,1.0):
        width=round(rw*fraction);height=round(width*qh/qw)
        if width<12 or height<12 or height>rh or width>rw:
            continue
        template=cv2.resize(query,(width,height),interpolation=cv2.INTER_AREA if width<qw else cv2.INTER_LINEAR)
        if template.std()<2:
            continue
        score=cv2.matchTemplate(reference,template,cv2.TM_CCOEFF_NORMED)
        best=max(best,float(score.max()))
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--only-unconfirmed',action='store_true')
    parser.add_argument('--min-score',type=float,default=-1.,help='Keep original order below this correlation; not a calibrated confidence threshold')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    rows=[]
    for report_path in args.reports:
        report=json.loads(report_path.read_text())
        for item in report['rows']:
            if not item.get('target'):
                continue
            query=gray(report_path.parent/item['query'])
            results=item['results']
            scores=[match(query,gray(frame_path(r))) for r in results]
            order=np.argsort(-np.array(scores))
            selected=int(order[0])
            if scores[selected]<args.min_score:
                selected=0
            if args.only_unconfirmed and results[0]['points']>=15:
                selected=0
            result=results[selected]
            target=item['target']
            correct=lambda r:f"{r['source']}/{r['filename']}"==target
            row={"report":str(report_path),"id":item['id'],"variant":item['variant'],
                 "target":target,"before_exact":correct(results[0]),"after_exact":correct(result),
                 "target_available":any(correct(r) for r in results),
                 "before":results[0],"after":result,
                 "template_candidates":[{"source":r['source'],"filename":r['filename'],"score":s}
                                        for r,s in zip(results,scores)]}
            rows.append(row)
            print(item['id'],item['variant'],row['before_exact'],'->',row['after_exact'],
                  result['filename'],f"NCC {scores[selected]:.3f}",flush=True)
            args.out.write_text(json.dumps({'notes':__doc__,'only_unconfirmed':args.only_unconfirmed,'min_score':args.min_score,'rows':rows},indent=2)+'\n')
    print(f"Exact: {sum(r['before_exact'] for r in rows)}/{len(rows)} -> {sum(r['after_exact'] for r in rows)}/{len(rows)}; "
          f"target available: {sum(r['target_available'] for r in rows)}/{len(rows)}",flush=True)


if __name__=='__main__':
    main()
