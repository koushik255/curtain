"""Movie-disjoint crop benchmark over unseen lbfive movies."""
import io, json, hashlib, random, time, tarfile, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from training.train import CurtainEncoder, reference_transform

CONDITIONS=('original','crop_center_0.9','crop_center_0.8','crop_topright_0.8','crop_center_0.65','combined')

def alter(image, condition):
    if condition=='original': return image
    w,h=image.size
    if condition=='combined':
        image=TF.adjust_gamma(image,1.2)
        image=image.resize((320,max(1,round(image.height*320/image.width))),Image.Resampling.BILINEAR)
        b=io.BytesIO(); image.save(b,format='JPEG',quality=40); b.seek(0)
        with Image.open(b) as x: return x.convert('RGB')
    scale={'crop_center_0.9':.9,'crop_center_0.8':.8,'crop_center_0.65':.65,'crop_topright_0.8':.8}[condition]
    cw,ch=round(w*scale),round(h*scale)
    x,y=((w-cw),0) if condition=='crop_topright_0.8' else ((w-cw)//2,(h-ch)//2)
    return image.crop((x,y,x+cw,y+ch))

@torch.inference_mode()
def encode(model, paths, size, device, condition='original'):
    out=[]
    for start in range(0,len(paths),256):
        images=[]
        for path in paths[start:start+256]:
            with Image.open(path) as opened:
                image=ImageOps.exif_transpose(opened).convert('RGB')
            images.append(reference_transform(size)(alter(image,condition)))
        with torch.autocast('cuda',dtype=torch.float16):
            batch=model(torch.stack(images).to(device)).float().cpu()
        out.append(batch)
    return torch.cat(out)

@torch.inference_mode()
def main(data_root, collection, checkpoints, splits, output):
    started=time.perf_counter(); device=torch.device('cuda'); random.seed(20260910)
    known={n.casefold() for key in ('training_movies','evaluation_movies') for n in splits[key]}
    movies=[m for m in collection['movies'] if m['movie'].casefold() not in known]
    reference_counts=[]
    with tempfile.TemporaryDirectory(prefix='unseen-benchmark-') as temp:
        temp=Path(temp); reference_paths=[]; queries=[]; targets=[]
        for movie in movies:
            folder=temp/movie['movie']; folder.mkdir()
            with tarfile.open(data_root/f"{movie['movie']}.tar") as archive:
                members=sorted((m for m in archive.getmembers() if m.isfile() and Path(m.name).suffix.lower() in {'.jpg','.jpeg','.png','.webp'}),key=lambda m:m.name)
                selected=sorted(random.sample(members,min(512,len(members))),key=lambda m:m.name)
                archive.extractall(folder,members=selected,filter='data')
            paths=[folder/member.name for member in selected]
            query_positions=sorted(random.sample(range(len(paths)),min(32,len(paths))))
            offset=len(reference_paths)
            reference_paths.extend(paths); queries.extend(paths[i] for i in query_positions)
            targets.extend(offset+i for i in query_positions); reference_counts.append(len(paths))
            print(f"Prepared {movie['movie']}: {len(paths)} references, {len(query_positions)} queries",flush=True)
        result=dict(seed=20260910, movie_count=len(movies), reference_frames=len(reference_paths),
                    query_frames=len(queries), conditions=list(CONDITIONS), movies=[m['movie'] for m in movies], models={},
                    caveat='All 112 movies are absent from the saved training and validation lists under case-insensitive matching. The fixed benchmark samples up to 512 references and 32 included query frames per movie; it is larger and more diverse than prior validation tests but is not the full production index.')
        target_tensor=torch.tensor(targets,device=device)
        ends=torch.tensor(np.cumsum(reference_counts),device=device)
        for model_name, checkpoint in checkpoints.items():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False); cfg=saved['config']
            model=CurtainEncoder(cfg['embedding_dim']).to(device).eval(); model.load_state_dict(saved['model'])
            references=encode(model,reference_paths,cfg['image_size'],device).to(device)
            metrics={}
            for condition in CONDITIONS:
                q=encode(model,queries,cfg['image_size'],device,condition).to(device)
                top=torch.cat([(q[i:i+32]@references.T).topk(5,dim=1).indices for i in range(0,len(q),32)])
                exact=top[:,0].eq(target_tensor); top5=top.eq(target_tensor[:,None]).any(1)
                pred_movie=torch.bucketize(top[:,0],ends); true_movie=torch.bucketize(target_tensor,ends)
                metrics[condition]=dict(top1=float(exact.float().mean()),top5=float(top5.float().mean()),movie_top1=float(pred_movie.eq(true_movie).float().mean()))
                print(json.dumps(dict(model=model_name,condition=condition,**metrics[condition])),flush=True)
            result['models'][model_name]=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),metrics=metrics)
    result['elapsed_seconds']=time.perf_counter()-started; Path(output).write_text(json.dumps(result,indent=2)+'\n'); print('COMPLETE',output,flush=True); return result
