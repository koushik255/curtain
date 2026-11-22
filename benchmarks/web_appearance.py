"""Reproducible image edits for screenshot retrieval experiments.

Fit observable output characteristics on development web images only, then
generate labelled variants from a separate indexed frame. Absolute web-image
brightness does not identify a colour transform; the optional aligned-pair
measurements distinguish the two. This tool does not change training defaults.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from benchmarks.online_benchmark import OUT, ROOT, digest, read, save


@dataclass(frozen=True)
class Edit:
    # Fractions of each original side retained, NOT fraction of area.
    width: float = 1.0
    height: float = 1.0
    x: float = .5
    y: float = .5
    gamma: float = 1.0
    contrast: float = 1.0
    lift: float = 0.0
    red: float = 1.0
    green: float = 1.0
    blue: float = 1.0
    saturation: float = 1.0
    blur: float = 0.0
    long_side: int = 640
    format: str = 'JPEG'
    quality: int = 85
    border: float = 0.0
    subtitle: str = ''
    player_controls: bool = False


def render(source: Image.Image, edit: Edit) -> tuple[Image.Image, dict]:
    """Return decoded edited pixels and an exact source-space crop label."""
    if not (0 < edit.width <= 1 and 0 < edit.height <= 1):
        raise ValueError('Crop side fractions must be in (0, 1]')
    if not (0 <= edit.x <= 1 and 0 <= edit.y <= 1):
        raise ValueError('Crop positions must be in [0, 1]')
    if edit.gamma <= 0 or edit.long_side < 16 or not 1 <= edit.quality <= 100:
        raise ValueError('Invalid gamma, output size or quality')
    if edit.format not in ('JPEG','WEBP','PNG') or not 0 <= edit.border <= .5 or edit.blur < 0:
        raise ValueError('Invalid format, border or blur')
    image=ImageOps.exif_transpose(source).convert('RGB')
    w,h=image.size;cw=max(1,round(w*edit.width));ch=max(1,round(h*edit.height))
    x=round((w-cw)*edit.x);y=round((h-ch)*edit.y)
    box=(x,y,x+cw,y+ch);image=image.crop(box)
    pixels=np.asarray(image,dtype=np.float32)/255
    pixels=np.power(pixels,edit.gamma)
    pixels=(pixels-.5)*edit.contrast+.5+edit.lift
    pixels*=np.array([edit.red,edit.green,edit.blue],dtype=np.float32)
    gray=pixels@np.array([.299,.587,.114],dtype=np.float32)
    pixels=gray[...,None]+edit.saturation*(pixels-gray[...,None])
    image=Image.fromarray(np.uint8(np.rint(np.clip(pixels*255,0,255))))
    if edit.blur:image=image.filter(ImageFilter.GaussianBlur(edit.blur))
    image.thumbnail((edit.long_side,edit.long_side),Image.Resampling.LANCZOS)
    if edit.border:
        by=round(image.height*edit.border)
        image=ImageOps.expand(image,border=(0,by),fill='black')
    draw=ImageDraw.Draw(image)
    if edit.subtitle:
        font_size=max(9,round(image.width*.028))
        font=ImageFont.load_default(size=font_size)
        draw.text((image.width/2,image.height*.9),edit.subtitle,anchor='mm',fill='white',
                  stroke_width=max(1,font_size//12),stroke_fill='black',font=font)
    if edit.player_controls:
        draw.rectangle((0,0,image.width,image.height*.08),fill=(28,28,28))
        draw.text((8,4),'II     CC                  settings',fill='white')
        draw.line((10,image.height-10,image.width-10,image.height-10),fill='white',width=2)
    buffer=io.BytesIO()
    options={} if edit.format=='PNG' else {'quality':edit.quality}
    image.save(buffer,format=edit.format,**options)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:result=decoded.convert('RGB')
    return result,{'edit':asdict(edit),'source_crop_xyxy':box,'retained_area':cw*ch/(w*h),
                   'output_size':result.size,'encoding_sha256':digest(buffer.getvalue())}


def quantiles(values):
    return np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist() if values else None


def profile(args):
    frozen=read(args.out/'frozen.json')
    provenance={r['id']:r['provenance'] for r in read(args.out/'source-provenance.json')['records']}
    queries=[q for q in frozen['queries'] if q['partition']=='development'
             and provenance[q['id']]=='documented_frame_capture']
    rows=[json.loads(line) for line in (args.out/'results.jsonl').read_text().splitlines()]
    by_id={r['id']:r for r in rows if r['partition']=='development'}
    if any(q['id'] not in by_id for q in queries):
        raise ValueError('Finish evaluating the documented development queries before fitting a profile')
    pairs=[]
    for q in queries:
        row=by_id.get(q['id'])
        if not row or not row['before_movie']:continue
        result=row['before']
        if result['points']<30 or not result['matrix']:continue
        with Image.open(result['path']) as im:reference=np.asarray(im.convert('RGB'))
        with Image.open(row['query']) as im:query=np.asarray(im.convert('RGB'))
        a,b,c,d,e,f=result['matrix'];matrix=np.float32([[a,c,e],[b,d,f]])
        qh,qw=query.shape[:2];rh,rw=reference.shape[:2]
        corners=cv2.transform(np.float32([[[0,0],[qw,0],[qw,qh],[0,qh]]]),matrix)[0]
        frame=np.float32([[0,0],[rw,0],[rw,rh],[0,rh]])
        area,_=cv2.intersectConvexConvex(corners,frame)
        coverage=float(area/(rw*rh))
        warped=cv2.warpAffine(reference,matrix,(qw,qh),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        valid=cv2.warpAffine(np.ones((rh,rw),np.uint8),matrix,(qw,qh),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
        src=warped[valid].astype(np.float32)/255;dst=query[valid].astype(np.float32)/255
        if len(src)<100:continue
        src_l=src@np.float32([.299,.587,.114]);dst_l=dst@np.float32([.299,.587,.114])
        # Percentile matching is descriptive, and tolerates small temporal differences.
        sx=np.quantile(src_l,[.1,.5,.9]);sy=np.quantile(dst_l,[.1,.5,.9])
        gain=float((sy[2]-sy[0])/max(.02,sx[2]-sx[0]));offset=float(sy[1]-gain*sx[1])
        pairs.append({'id':q['id'],'coverage':coverage,'luma_gain':gain,'luma_offset':offset,
                      'width_fraction':float(np.clip(np.ptp(corners[:,0])/rw,.02,1.)),
                      'height_fraction':float(np.clip(np.ptp(corners[:,1])/rh,.02,1.)),
                      'points':result['points'],'exact_frame_known':False})
    formats={f:sum(q['metadata']['format']==f for q in queries) for f in sorted({q['metadata']['format'] for q in queries})}
    result={'partition':'development','source_ids':[q['id'] for q in queries],
            'frozen_sha256':digest((args.out/'frozen.json').read_bytes()),'images':len(queries),
            'formats':formats,'long_side_quantiles':quantiles([max(q['metadata']['width'],q['metadata']['height']) for q in queries]),
            'saturation_quantiles':quantiles([q['metadata']['mean_saturation'] for q in queries]),
            'aligned_pairs':pairs,'coverage_quantiles':quantiles([p['coverage'] for p in pairs]),
            'luma_gain_quantiles':quantiles([p['luma_gain'] for p in pairs]),
            'luma_offset_quantiles':quantiles([p['luma_offset'] for p in pairs]),
            'limitations':'Documented frame-capture development sources only. Aligned measurements use successful same-movie SIFT matches and are biased against hard cases. Not exact-frame truth. Native dimensions describe web images; source codec and original crop are unknown. No test images used.'}
    save(args.out/'appearance-profile.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_ids','aligned_pairs')},indent=2))


def variants(args):
    frozen=read(args.out/'frozen.json')
    # Benchmark movies are excluded in full, including the development movies.
    held_out={q['movie'].casefold() for q in frozen['queries']}
    if args.movie.casefold() in held_out:
        raise ValueError('Choose a source outside all benchmark movies; benchmark originals are evaluation-only')
    profile_data=read(args.out/'appearance-profile.json')
    if profile_data['partition']!='development' or profile_data['frozen_sha256']!=digest((args.out/'frozen.json').read_bytes()):
        raise ValueError('Appearance profile must come from this benchmark development partition')
    rng=random.Random(args.seed)
    measured_sides=profile_data['long_side_quantiles']
    measured_side=int(min(640,max(160,rng.choice(measured_sides))))
    pair=rng.choice(profile_data['aligned_pairs']) if profile_data['aligned_pairs'] else {'coverage':1.,'luma_gain':1.,'luma_offset':0.}
    side=math.sqrt(max(.02,min(1.,pair['coverage'])))
    width=pair.get('width_fraction',side);height=pair.get('height_fraction',side)
    gain=float(np.clip(pair['luma_gain'],.65,1.4));offset=float(np.clip(pair['luma_offset'],-.15,.15))
    edits={
        'observed-dev':Edit(width=width,height=height,x=rng.random(),y=rng.random(),contrast=gain,
                            lift=offset+(gain-1)*.5,long_side=measured_side),
        # Stress presets are hypotheses, kept separate from the measured profile.
        'crop-grade-lowres':Edit(width=.4,height=.4,x=rng.random(),y=rng.random(),gamma=1.3,red=1.12,blue=.82,long_side=160,quality=35),
        'tiny-crop':Edit(width=.15,height=.15,x=rng.random(),y=rng.random(),long_side=160,quality=25),
        'player-square':Edit(width=.6,height=1.,gamma=1.12,long_side=360,player_controls=True,quality=45),
        'subtitles-webp':Edit(width=.75,height=.75,long_side=480,format='WEBP',quality=30,subtitle='Example subtitle'),
        'lifted-black-bars':Edit(contrast=.85,lift=.06,red=.9,blue=1.1,border=.12,quality=60),
        'blur-recompress':Edit(blur=1.2,long_side=240,quality=15),
    }
    if args.focus_x is not None:
        if not (0 <= args.focus_x <= 1 and 0 <= args.focus_y <= 1):
            raise ValueError('Focus coordinates must be fractions in [0, 1]')
        def position(center,fraction):
            return float(np.clip((center-fraction/2)/(1-fraction),0,1)) if fraction<1 else .5
        edits={name:replace(edit,x=position(args.focus_x,edit.width),y=position(args.focus_y,edit.height))
               for name,edit in edits.items()}
    args.destination.mkdir(parents=True,exist_ok=True)
    with Image.open(args.source) as image:
        original=ImageOps.exif_transpose(image).convert('RGB');original.save(args.destination/'original.png')
        ledger=[]
        for name,edit in edits.items():
            result,label=render(original,edit)
            # PNG preserves the already decoded lossy pixels without another re-encode.
            result.save(args.destination/(name+'.png'))
            ledger.append({'name':name,'file':name+'.png',**label})
    save(args.destination/'edits.json',{'movie':args.movie,'source':str(args.source.resolve()),
         'source_sha256':digest(args.source.read_bytes()),'seed':args.seed,
         'profile_sha256':digest((args.out/'appearance-profile.json').read_bytes()),
         'focus_xy':[args.focus_x,args.focus_y],
         'observed_pair':pair,'variants':ledger,
         'notes':'Label-preserving diagnostics only. Stress preset magnitudes are hypotheses, not inferred codec settings. Not integrated into training.'})
    preview=Image.new('RGB',(1280,560),'#202020');draw=ImageDraw.Draw(preview)
    for i,name in enumerate(['original',*edits]):
        x=i%4*320;y=i//4*280
        with Image.open(args.destination/(name+'.png')) as im:
            fitted=ImageOps.contain(im.convert('RGB'),(310,240))
        preview.paste(fitted,(x+(320-fitted.width)//2,y+30+(240-fitted.height)//2))
        draw.text((x+5,y+8),name,fill='white')
    preview.save(args.destination/'preview.jpg',quality=90)
    print('Saved',len(edits),'labelled variants to',args.destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['profile','variants'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--movie')
    parser.add_argument('--destination',type=Path,default=OUT/'appearance-lab')
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--focus-x',type=float,help='Optional subject center as fraction of source width')
    parser.add_argument('--focus-y',type=float,help='Optional subject center as fraction of source height')
    args=parser.parse_args()
    if args.command=='variants' and (not args.source or not args.movie):parser.error('variants requires --source and --movie')
    if (args.focus_x is None)!=(args.focus_y is None):parser.error('Specify both --focus-x and --focus-y')
    globals()[args.command](args)


if __name__=='__main__':main()
