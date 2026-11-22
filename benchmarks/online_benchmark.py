"""Collect and audit independently published scene screenshots.

The movie sample is fixed in benchmarks/data/online_movies_v1.json. Acquisition
and eligibility happen before retrieval. Failed downloads and ambiguous sources
remain in the coverage ledger. Exact frame IDs are never inferred as truth
from either retrieval system under comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/benchmarks/online-v1'
SAMPLE = ROOT / 'benchmarks/data/online_movies_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    url=urllib.parse.quote(url,safe=":/?=&%#@+;,!~*'()")
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; Curtain screenshot benchmark)'})
    with urllib.request.urlopen(request,timeout=20) as response:
        return response.read()


class Gallery(HTMLParser):
    def __init__(self):
        super().__init__();self.images=[];self.text=[]

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='a' and 'bwg_lightbox' in attrs.get('class',''):
            self.images.append(attrs['href'])

    def handle_data(self,text):
        self.text.append(text)


def image_metadata(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        format=image.format
        rgb=ImageOps.exif_transpose(image).convert('RGB')
        width,height=rgb.size
        thumb=np.asarray(rgb.resize((9,8)).convert('L'))
        bits=np.packbits((thumb[:,1:]>thumb[:,:-1]).reshape(-1)).tobytes().hex()
        pixels=np.asarray(rgb.resize((256,256)),dtype=np.float32)/255
        luma=pixels@np.array([.299,.587,.114],dtype=np.float32)
        sat=(pixels.max(axis=-1)-pixels.min(axis=-1))/np.maximum(pixels.max(axis=-1),1/255)
        tables=getattr(image,'quantization',None)
        return {'format':format,'width':width,'height':height,'aspect':width/height,
                'dhash':bits,'luma_quantiles':np.quantile(luma,[.01,.1,.5,.9,.99]).tolist(),
                'mean_saturation':float(sat.mean()),'jpeg_quantization':tables,
                'sha256':digest(data),'bytes':len(data)}


def collect_movie(case, source, out, seed):
    movie=case['movie'];originals=out/'originals';originals.mkdir(exist_ok=True)
    pages=out/'pages';pages.mkdir(exist_ok=True)
    ledger={'movie':movie,'partition':case['partition'],'requested':5,'attempts':[],'queries':[]}
    if source is None:
        ledger['error']='No independently sourced gallery/candidate pool discovered'
        return ledger
    try:
        if source.get('candidates') is not None:
            candidates=list(source['candidates'])
        else:
            page=pages/(movie+'.html')
            data=page.read_bytes() if page.exists() else fetch(source['gallery_url'])
            page.write_bytes(data)
            gallery=Gallery();gallery.feed(data.decode('utf-8',errors='replace'))
            text=' '.join(gallery.text)
            year=movie.rsplit('_',1)[1]
            if not re.search(r'Year:\s*'+year+r'\b',text):
                raise ValueError(f'Gallery year could not be verified as {year}')
            urls=list(dict.fromkeys(gallery.images))
            if len(urls)<5:
                raise ValueError(f'Only {len(urls)} gallery images parsed')
            candidates=[{'page_url':source['gallery_url'],'image_url':url,
                         'source_kind':'dedicated_frame_gallery','source_assertion':'Filmgrab frame gallery, movie title matched and year checked'} for url in urls]
            ledger['gallery_sha256']=digest(data)
        # Persist the complete discovered pool, before any search evaluation.
        ledger['candidate_pool']=candidates
        rng=random.Random(f'{seed}/{movie}')
        order=list(range(len(candidates)));rng.shuffle(order)
        ledger['candidate_order']=order
        used=[]
        for position in order:
            candidate=candidates[position]
            attempt={**candidate,'pool_position':position}
            try:
                key=f'{movie}__{len(ledger["queries"]):02d}'
                cache=out/'download-cache'/digest(candidate['image_url'].encode())
                cache.parent.mkdir(exist_ok=True)
                data=cache.read_bytes() if cache.exists() else fetch(candidate['image_url'])
                meta=image_metadata(data)
                if min(meta['width'],meta['height'])<80:
                    raise ValueError('Image too small to distinguish a scene from an icon/thumbnail')
                duplicate=next((old for old in used if old['sha256']==meta['sha256'] or
                                (int(old['dhash'],16)^int(meta['dhash'],16)).bit_count()<=2),None)
                if duplicate:
                    attempt['excluded']='Exact or near-duplicate of another selected image'
                    attempt['duplicate_of']=duplicate['id']
                else:
                    cache.write_bytes(data)
                    (originals/key).write_bytes(data)
                    query={**case,**candidate,'id':key,'pool_position':position,'metadata':meta,
                           'visual_review':'pending','eligibility':'source_asserted_scene'}
                    ledger['queries'].append(query);used.append({**meta,'id':key})
                    attempt['selected']=key
            except (OSError,ValueError) as exc:
                attempt['error']=str(exc)
            ledger['attempts'].append(attempt)
            if len(ledger['queries'])==5:
                break
    except (OSError,ValueError) as exc:
        ledger['error']=str(exc)
    return ledger


def collect(args):
    sample=read(args.sample);sources={s['movie']:s for s in read(args.sources)}
    args.out.mkdir(parents=True,exist_ok=True)
    ledgers=args.out/'acquisition';ledgers.mkdir(exist_ok=True)
    cases=[c for c in sample['movies'] if not args.ids or c['movie'] in args.ids]
    with ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(collect_movie,c,sources.get(c['movie']),args.out,sample['seed']):c for c in cases}
        for future in as_completed(jobs):
            case=jobs[future]
            try:result=future.result()
            except Exception as exc:result={**case,'requested':5,'queries':[],'error':repr(exc)}
            save(ledgers/(case['movie']+'.json'),result)
            print(f"{case['movie']}: {len(result['queries'])}/5 {result.get('error','')}",flush=True)
    assemble(args.out,args.sample)


def assemble(out,sample_path):
    sample=read(sample_path);queries=[];coverage=[]
    for case in sample['movies']:
        path=out/'acquisition'/(case['movie']+'.json')
        ledger=read(path) if path.exists() else {'queries':[],'error':'Not yet attempted'}
        queries.extend(ledger['queries'])
        coverage.append({**case,'requested':5,'acquired':len(ledger['queries']),
                         'error':ledger.get('error'),'attempted':len(ledger.get('attempts',[]))})
    manifest={'version':1,'seed':sample['seed'],'sampling':sample['sampling'],
              'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'sample_sha256':digest(Path(sample_path).read_bytes()),'coverage':coverage,'queries':queries,
              'notes':'Independent web sources. Acquisition order and eligibility are decided before model results. Exact frame IDs unknown until separately reviewed. No scene labels inferred from retrieval success. Development/test movies fixed before acquisition.'}
    save(out/'manifest.json',manifest)
    print(f"Acquired {len(queries)}/{len(sample['movies'])*5} images; {sum(c['acquired']==5 for c in coverage)}/{len(coverage)} movies complete",flush=True)


def sheets(args):
    manifest=read(args.out/'manifest.json');items=manifest['queries']
    directory=args.out/'source-sheets';directory.mkdir(exist_ok=True)
    for start in range(0,len(items),15):
        group=items[start:start+15]
        canvas=Image.new('RGB',(1500,260*((len(group)+2)//3)),'#202020');draw=ImageDraw.Draw(canvas)
        for i,q in enumerate(group):
            x=i%3*500;y=i//3*260
            with Image.open(args.out/'originals'/q['id']) as im:
                fitted=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(490,220))
            canvas.paste(fitted,(x,y+23));draw.text((x,y+3),q['id'],fill='white')
            draw.text((x,y+245),q['metadata']['format']+' '+str(q['metadata']['width'])+'x'+str(q['metadata']['height']),fill='white')
        canvas.save(directory/f'{start//15:03d}.jpg',quality=90)
    save(directory/'index.json',[[q['id'] for q in items[i:i+15]] for i in range(0,len(items),15)])
    print(f'{(len(items)+14)//15} source sheets saved',flush=True)


def eligible_queries(out):
    manifest=read(out/'manifest.json');reviews=read(out/'reviews.json')
    return [q for q in manifest['queries'] if reviews.get(q['id'],{}).get('sha256')==q['metadata']['sha256']
            and reviews[q['id']]['status']=='eligible']


def restore(args):
    """Restore exactly the recorded query bytes, rejecting changed remote assets."""
    snapshot=read(args.queries)
    if (args.out/'frozen.json').exists():raise FileExistsError('Restore into a new output directory')
    (args.out/'originals').mkdir(parents=True,exist_ok=True)
    def one(q):
        path=args.out/'originals'/q['id']
        try:
            data=path.read_bytes() if path.exists() else fetch(q['image_url'])
            if digest(data)!=q['metadata']['sha256']:raise ValueError('Published image bytes changed; do not substitute silently')
            path.write_bytes(data)
            return {'id':q['id'],'restored':True}
        except (OSError,ValueError) as exc:return {'id':q['id'],'error':str(exc)}
    with ThreadPoolExecutor(args.workers) as pool:ledger=list(pool.map(one,snapshot['frozen']['queries']))
    save(args.out/'restore-ledger.json',ledger)
    if any('error' in r for r in ledger):raise RuntimeError('Some sources could not be restored; see restore-ledger.json')
    save(args.out/'frozen.json',snapshot['frozen'])
    save(args.out/'source-provenance.json',snapshot['source_provenance'])
    save(args.out/'reviews.json',snapshot['source_reviews'])
    print('Restored',len(ledger),'checksum-verified web images')


def refill(args):
    """Use the pre-shuffled reserves after source review, before retrieval."""
    reviews=read(args.out/'reviews.json')
    def one(case):
        movie=case['movie'];path=args.out/'acquisition'/(movie+'.json');ledger=read(path)
        rejected=ledger.setdefault('rejected_queries',[])
        keep=[];vacant=[]
        for q in ledger['queries']:
            review=reviews.get(q['id'],{})
            if review.get('sha256')==q['metadata']['sha256'] and review.get('status')=='excluded':
                rejected.append({**q,'review':review});vacant.append(q['id'])
                archive=args.out/'rejected-originals';archive.mkdir(exist_ok=True)
                (archive/q['metadata']['sha256']).write_bytes((args.out/'originals'/q['id']).read_bytes())
            else:keep.append(q)
        occupied={q['id'] for q in keep}
        vacant=sorted(set(vacant)|{f'{movie}__{i:02d}' for i in range(5) if f'{movie}__{i:02d}' not in occupied})
        attempted={a['image_url'] for a in ledger['attempts']}
        for position in ledger['candidate_order']:
            if not vacant:break
            c=ledger['candidate_pool'][position]
            if c['image_url'] in attempted:continue
            attempt={**c,'pool_position':position,'phase':'source_review_refill'}
            try:
                cache=args.out/'download-cache'/digest(c['image_url'].encode())
                data=cache.read_bytes() if cache.exists() else fetch(c['image_url'])
                meta=image_metadata(data)
                if min(meta['width'],meta['height'])<80:raise ValueError('Image too small')
                if any(meta['sha256']==q['metadata']['sha256'] or
                       (int(meta['dhash'],16)^int(q['metadata']['dhash'],16)).bit_count()<=2 for q in keep):
                    attempt['excluded']='Duplicate of another selected image'
                else:
                    key=vacant.pop(0);cache.write_bytes(data);(args.out/'originals'/key).write_bytes(data)
                    keep.append({**case,**c,'id':key,'pool_position':position,'metadata':meta,
                                 'visual_review':'pending','eligibility':'source_asserted_scene'})
                    attempt['selected']=key
            except (OSError,ValueError) as exc:attempt['error']=str(exc)
            ledger['attempts'].append(attempt)
        ledger['queries']=sorted(keep,key=lambda q:q['id']);save(path,ledger)
        return movie,len(keep),len(vacant)
    cases=read(args.sample)['movies']
    if args.ids:cases=[c for c in cases if c['movie'] in args.ids]
    with ThreadPoolExecutor(args.workers) as pool:
        for movie,count,missing in pool.map(one,cases):
            print(movie,count,'missing',missing,flush=True)
    assemble(args.out,args.sample)


def freeze(args):
    manifest=read(args.out/'manifest.json');queries=eligible_queries(args.out)
    reviews=read(args.out/'reviews.json')
    pending=[q['id'] for q in manifest['queries'] if reviews.get(q['id'],{}).get('sha256')!=q['metadata']['sha256']
             or reviews[q['id']].get('status') not in ('eligible','excluded')]
    if pending:raise ValueError(f'{len(pending)} images still require review')
    config={'version':1,'manifest_sha256':digest((args.out/'manifest.json').read_bytes()),
            'reviews_sha256':digest((args.out/'reviews.json').read_bytes()),'queries':queries,
            'reranker':{'name':'multiscale_grayscale_ncc','min_score':.95,'candidates':'same returned 20',
                        'implementation_sha256':digest((ROOT/'benchmarks/template_rerank.py').read_bytes())},
            'query_processing':{'long_side':640,'format':'JPEG','quality':90},
            'exact_frame_truth':'unknown; never inferred from either compared retrieval method',
            'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    path=args.out/'frozen.json'
    if path.exists():raise FileExistsError('Frozen benchmark already exists; use a new output directory to change it')
    save(path,config);print('Frozen',len(queries),'eligible queries',flush=True)


def run(args):
    # Set the index before import: search.load() binds its default at import time.
    os.environ['CURTAIN_FAISS_DIR']=str(args.index.resolve())
    import cv2
    import torch
    from apps.server import TrainedSearchEngine
    from rcmindex.search import EXCLUSIONS_FILE
    from benchmarks.template_rerank import match
    torch.set_num_threads(args.threads);cv2.setNumThreads(1)
    config=read(args.out/'frozen.json')
    if digest((ROOT/'benchmarks/template_rerank.py').read_bytes())!=config['reranker']['implementation_sha256']:
        raise ValueError('Reranker implementation changed after freeze')
    provenance={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint.read_bytes()),
                'exclusions_sha256':digest(EXCLUSIONS_FILE.read_bytes()),
                'index':str(args.index.resolve()),'frozen_sha256':digest((args.out/'frozen.json').read_bytes()),
                'index_catalog_sha256':digest((args.index/'catalog.json').read_bytes())}
    provenance_path=args.out/'run-config.json'
    if provenance_path.exists() and read(provenance_path)!=provenance:raise ValueError('Resume configuration differs')
    save(provenance_path,provenance)
    results_path=args.out/'results.jsonl'
    done={json.loads(line)['id'] for line in results_path.read_text().splitlines()} if results_path.exists() else set()
    engine=TrainedSearchEngine(args.checkpoint)
    queries=config['queries']
    if args.ids:queries=[q for q in queries if q['id'] in args.ids or q['movie'] in args.ids]
    query_dir=args.out/'queries';query_dir.mkdir(exist_ok=True)
    def record(entry):
        index,score,alignment=entry
        return {**engine.records[index],'index':int(index),'score':float(score),
                'points':int(alignment[0]) if alignment else 0,
                'matrix':alignment[1] if alignment else None,'path':str(engine.frame_path(index))}
    with results_path.open('a') as output:
        for i,q in enumerate(queries):
            if q['id'] in done:continue
            raw=(args.out/'originals'/q['id']).read_bytes()
            if digest(raw)!=q['metadata']['sha256']:raise ValueError('Source checksum changed: '+q['id'])
            started=time.perf_counter()
            with Image.open(io.BytesIO(raw)) as source:
                im=ImageOps.exif_transpose(source).convert('RGB');im.thumbnail((640,640))
            buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=90)
            query_path=query_dir/(q['id']+'.jpg');query_path.write_bytes(buffer.getvalue())
            with Image.open(io.BytesIO(buffer.getvalue())) as decoded:image=decoded.convert('RGB')
            readings=engine.embed_query(image);ranked,_=engine.rank(image,readings)
            baseline_ms=(time.perf_counter()-started)*1000
            candidates=[record(entry) for entry in ranked]
            gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
            for candidate in candidates:
                path=Path(candidate['path'])
                with Image.open(path) as reference:
                    candidate['ncc']=match(gray,np.asarray(reference.convert('L')))
            best=max(range(len(candidates)),key=lambda j:candidates[j]['ncc']) if candidates else None
            chosen=best if best is not None and candidates[best]['ncc']>=config['reranker']['min_score'] else 0
            before=candidates[0] if candidates else None;after=candidates[chosen] if candidates else None
            same=lambda r:r is not None and r['movie'].casefold()==q['movie'].casefold()
            row={'id':q['id'],'movie':q['movie'],'partition':q['partition'],'query':str(query_path),
                 'source_sha256':q['metadata']['sha256'],'query_sha256':digest(buffer.getvalue()),
                 'training_status':engine.movie_lookup[q['movie']]['training_status'],
                 'before':before,'after':after,'before_movie':same(before),'after_movie':same(after),
                 'movie_in_candidates':any(same(r) for r in candidates),'candidates':candidates,
                 'baseline_ms':baseline_ms,'reranker_ms':(time.perf_counter()-started)*1000-baseline_ms,
                 'before_exact':None,'after_exact':None,'exact_target':None}
            output.write(json.dumps(row)+'\n');output.flush()
            print(f"{i+1}/{len(queries)} {q['id']} movie {row['before_movie']} -> {row['after_movie']} "
                  f"NCC {after['ncc'] if after else -1:.3f} {baseline_ms:.0f}ms",flush=True)


def report(args):
    rows=[json.loads(line) for line in (args.out/'results.jsonl').read_text().splitlines()]
    frozen=read(args.out/'frozen.json')
    provenance_path=args.out/'source-provenance.json'
    provenance={r['id']:r['provenance'] for r in read(provenance_path)['records']} if provenance_path.exists() else {}
    summary={'complete':len(rows)==len(frozen['queries']),'evaluated':len(rows),'expected':len(frozen['queries']),
             'notes':'Movie identity only. Exact frame labels unknown. Unverified scene stills are not exact-match positives. Latencies are indicative CPU timings under concurrent local work, not a controlled performance benchmark.'}
    groups={partition:[r for r in rows if partition=='all' or r['partition']==partition] for partition in ('development','test','all')}
    for kind in sorted(set(provenance.values())):
        for partition in ('development','test','all'):
            groups[kind+'/'+partition]=[r for r in rows if provenance.get(r['id'])==kind and (partition=='all' or r['partition']==partition)]
    for partition,group in groups.items():
        summary[partition]={'images':len(group),'movies':len({r['movie'] for r in group}),
                            'baseline_movie_top1':sum(r['before_movie'] for r in group),
                            'reranked_movie_top1':sum(r['after_movie'] for r in group),
                            'movie_in_returned_candidates':sum(r['movie_in_candidates'] for r in group),
                            'changed_top1':sum(r['before']!=r['after'] for r in group),
                            'wrong_movie_with_15_sift_inliers':sum(not r['before_movie'] and r['before']['points']>=15 for r in group),
                            'wrong_movie_with_no_movie_candidate':sum(not r['before_movie'] and not r['movie_in_candidates'] for r in group),
                            'movie_regressions':sum(r['before_movie'] and not r['after_movie'] for r in group),
                            'movie_recoveries':sum(not r['before_movie'] and r['after_movie'] for r in group),
                            'training_status':{status:sum(r['training_status']==status for r in group) for status in sorted({r['training_status'] for r in group})},
                            'baseline_ms_median':float(np.median([r['baseline_ms'] for r in group])) if group else None,
                            'reranker_ms_median':float(np.median([r['reranker_ms'] for r in group])) if group else None,
                            'exact_frame_accuracy':None}
    save(args.out/'summary.json',summary);print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['collect','assemble','sheets','refill','freeze','run','report','restore'])
    parser.add_argument('--sample',type=Path,default=SAMPLE)
    parser.add_argument('--sources',type=Path,default=OUT/'sources.json')
    parser.add_argument('--queries',type=Path,default=ROOT/'benchmarks/data/online_queries_v1.json')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'curtain_models/all-films-v1/best.pt')
    parser.add_argument('--index',type=Path,default=ROOT/'.local/faiss-all-films-v1')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.command=='collect':collect(args)
    elif args.command=='assemble':assemble(args.out,args.sample)
    else:globals()[args.command](args)


if __name__=='__main__':
    main()
