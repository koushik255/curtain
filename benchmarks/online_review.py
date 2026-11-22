"""Build local, inspectable query/result reports; image labels stay unknown."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmarks.online_benchmark import OUT, read, save


def panel(canvas,draw,path,x,y,label):
    draw.text((x+5,y+4),label[:91],fill='white')
    with Image.open(path) as im:
        fitted=ImageOps.contain(im.convert('RGB'),(490,215))
    canvas.paste(fitted,(x+(500-fitted.width)//2,y+28+(215-fitted.height)//2))


def sheets(args,rows):
    if args.kind=='changes':rows=[r for r in rows if r['before']['index']!=r['after']['index']]
    elif args.kind=='attention':rows=[r for r in rows if not r['before_movie'] or r['before']['points']<15]
    directory=args.out/f'result-sheets-{args.kind}';directory.mkdir(exist_ok=True)
    mapping=[]
    for start in range(0,len(rows),5):
        group=rows[start:start+5]
        canvas=Image.new('RGB',(1500,260*len(group)),'#202020');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(group):
            panel(canvas,draw,row['query'],0,i*260,row['id'])
            choices=[('before',row['before']),('after',row['after'])]
            if args.blind:random.Random('blind/'+row['id']).shuffle(choices)
            mapping.append({'id':row['id'],'A':choices[0][0],'B':choices[1][0],
                            'A_frame':choices[0][1]['source']+'/'+choices[0][1]['filename'],
                            'B_frame':choices[1][1]['source']+'/'+choices[1][1]['filename']})
            for j,(name,result) in enumerate(choices,1):
                label=('A' if j==1 else 'B') if args.blind else f"{name}: {result['movie']} {result['filename']} SIFT {result['points']} NCC {result['ncc']:.3f}"
                panel(canvas,draw,result['path'],j*500,i*260,label)
        canvas.save(directory/f'{start//5:03d}.jpg',quality=90)
    save(directory/'mapping.json',mapping)
    print(len(rows),'comparisons in',directory)


def report(args,rows):
    frozen=read(args.out/'frozen.json');source={q['id']:q for q in frozen['queries']}
    provenance={r['id']:r['provenance'] for r in read(args.out/'source-provenance.json')['records']}
    data=[]
    for row in rows:
        q=source[row['id']]
        data.append({'id':row['id'],'movie':row['movie'],'partition':row['partition'],
                     'before_movie':row['before_movie'],'after_movie':row['after_movie'],
                     'changed':row['before']['index']!=row['after']['index'],'provenance':provenance[row['id']],
                     'weak':row['before']['points']<15,'source':q['page_url'],
                     'query':os.path.relpath(row['query'],args.out),
                     **{name:{**row[name],'path':os.path.relpath(row[name]['path'],args.out)} for name in ('before','after')}})
    payload=json.dumps(data).replace('<','\\u003c')
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Curtain independent web benchmark</title>
<style>body{background:#17191c;color:#eee;font:15px system-ui;margin:24px}a{color:#a7cfff}header{position:sticky;top:0;background:#17191c;padding:12px 0;z-index:1}select,input,button{padding:8px;margin:4px}article{border-top:1px solid #555;padding:20px 0}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.images img{width:100%;height:250px;object-fit:contain;background:#080808}small{color:#bbb}figure{margin:0}figcaption{min-height:60px}@media(max-width:800px){.images{grid-template-columns:1fr}}</style>
<h1>Curtain: independent web images / 95 movies</h1><p>The collection contains 475 scene images: 319 documented frame captures and 156 unverified scene stills. Unverified stills are kept as a separate discovery cohort and are not exact-match positives. The current search and fixed NCC prototype use the same 20 candidates. Exact-frame accuracy is unknown. SIFT inliers and NCC are evidence, not truth labels. Read <a href="summary.json">metrics</a>, <a href="frozen.json">frozen sources/configuration</a>, <a href="source-provenance.json">source provenance</a>, and <a href="catalog-audit.json">catalog findings</a>.</p>
<header><input id="query" placeholder="Filter movie or image ID"><select id="provenance"><option value="documented_frame_capture">Documented screenshots</option><option value="all">All collected images</option><option value="scene_image_unverified">Unverified scene stills</option></select><select id="partition"><option value="all">Both partitions</option><option>development</option><option>test</option></select><select id="filter"><option value="all">All cases</option><option value="wrong">Wrong movie before</option><option value="changes">Reranker changed first result</option><option value="weak">Under 15 SIFT inliers</option></select><button id="export">Export my scene reviews</button><span id="count"></span></header>
<p>Review scene correspondence separately from exact timestamp. Reviews are stored in this browser and can be exported; they do not silently become benchmark ground truth.</p><main></main>
<script>const rows=PAYLOAD;const reviews=JSON.parse(localStorage.getItem('curtain-online-v1-scene-review')||'{}');
const make=(tag,text)=>{let el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;};
function render(){let items=rows.filter(r=>(document.querySelector('#provenance').value==='all'||r.provenance===document.querySelector('#provenance').value)&&(document.querySelector('#partition').value==='all'||r.partition===document.querySelector('#partition').value)&&r.id.toLowerCase().includes(document.querySelector('#query').value.toLowerCase()));
const filter=document.querySelector('#filter').value;items=items.filter(r=>filter==='all'||filter==='wrong'&&!r.before_movie||filter==='changes'&&r.changed||filter==='weak'&&r.weak);document.querySelector('#count').textContent=items.length+' images';const main=document.querySelector('main');main.replaceChildren();
for(const r of items){let card=make('article');card.append(make('h2',r.id),make('small',r.partition+' · '+r.provenance+' · movie before: '+r.before_movie+' · movie after: '+r.after_movie));let source=make('a','Original source page');source.href=r.source;source.target='_blank';card.append(make('p'),source);let grid=make('div');grid.className='images';
for(const key of ['query','before','after']){let fig=make('figure'),a=make('a'),img=make('img');a.href=key==='query'?r.query:r[key].path;a.target='_blank';img.src=a.href;img.loading='lazy';a.append(img);fig.append(a);fig.append(make('figcaption',key==='query'?'Web screenshot':key+': '+r[key].movie+' / '+r[key].filename+' · SIFT '+r[key].points+' · NCC '+r[key].ncc.toFixed(3)));grid.append(fig);}card.append(grid);
for(const key of ['before','after']){const label=make('label',key+' scene review: '),select=make('select');for(const value of ['unreviewed','same scene / composition','different scene','unclear']){const option=make('option',value);option.value=value;select.append(option);}select.value=reviews[r.id]?.[key]||'unreviewed';select.onchange=()=>{reviews[r.id]={...(reviews[r.id]||{}),[key]:select.value};localStorage.setItem('curtain-online-v1-scene-review',JSON.stringify(reviews));};label.append(select);card.append(label);}main.append(card);}}
for(const id of ['query','provenance','partition','filter'])document.querySelector('#'+id).oninput=render;
document.querySelector('#export').onclick=()=>{const a=make('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({kind:'manual scene correspondence; exact frame unknown',reviews},null,2)],{type:'application/json'}));a.download='scene-reviews.json';a.click();URL.revokeObjectURL(a.href);};render();</script></html>'''
    (args.out/'review.html').write_text(template.replace('PAYLOAD',payload))
    print('Saved',args.out/'review.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['report','sheets'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--kind',choices=['all','attention','changes'],default='attention')
    parser.add_argument('--blind',action='store_true')
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.out/'results.jsonl').read_text().splitlines()]
    globals()[args.command](args,rows)


if __name__=='__main__':main()
