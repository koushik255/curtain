"""CPU temporal head on frozen per-frame composition features.

Self-supervision: feature-noise invariance, ordered visual-change prediction,
and coarse optical-flow distillation. No camera-angle or auteur labels exist.
Film-held-out diagnostics measure proxy tasks, not directing-style accuracy.
The model consumes measured flow as well as ordered visual features.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

import composition_model as cm
import kurosawa_experiment as pilot
from focused_experiment import TARGETS
from sequence_experiment import gif_pair, VERSION as BASE_VERSION

BASE = cm.HERE/'results/sequence-v1'
OUT = cm.HERE/'results/temporal-v1'
HELD = {'Star_Wars_1977', 'Once_Upon_a_Time_in_the_West_1968', 'Harakiri_1962'}


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(4, 96)*.02)
        self.input = nn.Linear(128, 96)
        self.motion_input = nn.Sequential(nn.Linear(32,96),nn.GELU(),nn.Linear(96,96))
        layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.embedding = nn.Sequential(nn.LayerNorm(768), nn.Linear(768,128))
        self.flow = nn.Linear(128,96)
        self.change = nn.Linear(128,384)

    def forward(self, x, flow=None):
        if flow is None: flow=x.new_zeros((len(x),96))
        motion=self.motion_input(flow.reshape(-1,3,32)/3)
        motion=torch.cat([motion.new_zeros((len(x),1,96)),motion],dim=1)
        h = self.encoder(self.input(x)+motion+self.pos)
        changes=F.normalize(x[:,1:]-x[:,:-1],dim=-1).flatten(1)
        return F.normalize(self.embedding(torch.cat([h.flatten(1),changes],dim=1)),dim=-1)


def extract(args):
    OUT.mkdir(parents=True, exist_ok=True)
    clips=json.loads((BASE/'clips.json').read_text())
    names=list(dict.fromkeys(p for c in clips for p in c['paths']))
    signature=hashlib.sha256(json.dumps([(p,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in names]).encode()+(cm.OUT/'head.pt').read_bytes()).hexdigest()
    cache=OUT/f'frames-{signature}'
    cache.mkdir(exist_ok=True)
    head=cm.CompositionHead()
    head.load_state_dict(torch.load(cm.OUT/'head.pt',map_location='cpu',weights_only=True)); head.eval()
    backbone=pilot.load_model(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main',Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
    def load(path):
        with Image.open(path) as im: return cm.to_tensor(im.convert('RGB'))
    chunks=[]
    with ThreadPoolExecutor(6) as pool, torch.inference_mode():
        for start in range(0,len(names),32):
            path=cache/f'{start:06}.npy'
            if path.exists():
                chunks.append(np.load(path)); continue
            x=torch.stack(list(pool.map(load,names[start:start+32])))
            p=backbone.forward_features(x)['x_norm_patchtokens'].float()
            grid=p.reshape(-1,4,4,4,4,384).mean(dim=(2,4)).reshape(-1,16,384)
            z=head(grid).numpy()
            temporary=path.with_suffix('.partial.npy')
            np.save(temporary,z); temporary.replace(path); chunks.append(z)
            if start%320==0: pilot.note(f'frame features {start}/{len(names)}')
    z=np.concatenate(chunks); index={p:i for i,p in enumerate(names)}
    sequences=np.stack([z[[index[p] for p in c['paths']]] for c in clips])
    # Match the baseline cache signature, including clip order and file stats.
    motion_signature=hashlib.sha256(json.dumps([BASE_VERSION, [(c['frames'], [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in c['paths']]) for c in clips]]).encode()).hexdigest()
    caches=[BASE/f'motion-{motion_signature}.npz']
    if not caches[0].exists(): raise ValueError('Baseline motion cache does not match clips')
    with np.load(caches[0]) as data:
        flow=data['vectors'].copy(); energy=data['energy'].copy()
    if len(flow)!=len(clips): raise ValueError('Baseline motion rows do not align')
    np.savez(OUT/'features.npz',sequences=sequences,flow=flow,energy=energy)
    pilot.write_json(OUT/'clips.json',clips)
    pilot.write_json(OUT/'feature-manifest.json',dict(signature=signature,clips=len(clips),unique_frames=len(names),motion_cache=caches[0].name))


def augment(x):
    # Feature perturbations preserve ordering. Appearance robustness comes from
    # the frozen composition head; these are not image-level color augmentations.
    mask=(torch.rand(x.shape[0],1,x.shape[2])>.15).float()
    return F.normalize(x*mask+torch.randn_like(x)*.008,dim=-1)


def augment_flow(flow):
    mask=(torch.rand_like(flow)>.2).float()
    return flow*mask+torch.randn_like(flow)*.05


def reverse_flow(flow):
    return -flow.reshape(-1,3,32).flip(1).reshape(-1,96)


def split(clips):
    va=np.array([i for i,c in enumerate(clips) if c['movie'].split('/')[-1] in HELD])
    tr=np.array([i for i,c in enumerate(clips) if c['movie'].split('/')[-1] not in HELD])
    assert not {clips[i]['movie'] for i in tr}&{clips[i]['movie'] for i in va}
    return tr,va


def train(args):
    torch.manual_seed(7)
    data=np.load(OUT/'features.npz'); x=torch.from_numpy(data['sequences']).float()
    flow=torch.from_numpy(data['flow']).float()
    energy=torch.from_numpy(data['energy']).float()
    clips=json.loads((OUT/'clips.json').read_text()); tr,va=split(clips)
    # Statistics exclusively from training films.
    scale=flow[tr].std(0).clamp(min=.1)
    target=(flow/scale).clamp(-5,5)
    model=TemporalHead(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
    def diagnostics(ids):
        model.eval()
        with torch.no_grad():
            z=model(x[ids],flow[ids]); other=model(augment(x[ids]),augment_flow(flow[ids])); reverse=model(x[ids].flip(1),reverse_flow(flow[ids]))
            retrieval=(z@other.T).argmax(1).eq(torch.arange(len(ids))).float().mean().item()
            active=energy[ids]>=.002
            gap=((z*other).sum(1)-(z*reverse).sum(1))[active].mean().item()
            pred=model.flow(z)
            mse=F.mse_loss(pred,target[ids]).item()
            zero=target[ids].square().mean().item()
            mean_baseline=(target[ids]-target[tr].mean(0)).square().mean().item()
            return dict(augmented_clip_top1=retrieval,forward_vs_reverse_gap=gap,flow_mse=mse,zero_flow_mse=zero,train_mean_flow_mse=mean_baseline)
    torch.manual_seed(100); initial=diagnostics(va)
    log=[]; best=float('inf')
    for epoch in range(args.epochs):
        model.train(); losses=[]
        order=np.random.default_rng(epoch).permutation(tr)
        for start in range(0,len(order),args.batch):
            ids=order[start:start+args.batch]
            a,b=augment(x[ids]),augment(x[ids]); za,zb=model(a,augment_flow(flow[ids])),model(b,augment_flow(flow[ids]))
            logits=za@zb.T/.1; labels=torch.arange(len(ids))
            contrast=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
            motion=F.mse_loss(model.flow(za),target[ids])
            changes=F.normalize((x[ids,1:]-x[ids,:-1]).flatten(1),dim=-1)
            change=1-F.cosine_similarity(model.change(za),changes).mean()
            reverse=model(a.flip(1),reverse_flow(flow[ids]))
            active=energy[ids]>=.002
            margin=F.relu(.15+(za*reverse).sum(1)-(za*zb).sum(1))
            directional=margin[active].mean() if active.any() else margin.sum()*0
            loss=contrast+2*motion+.3*change+directional
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        torch.manual_seed(100); metrics=diagnostics(va)
        item=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),**metrics); log.append(item)
        # Select on held-out proxy flow regression, not on the displayed matches.
        if metrics['flow_mse']<best:
            best=metrics['flow_mse']; torch.save(dict(state_dict=model.state_dict(),epoch=epoch+1,flow_scale=scale),OUT/'head.pt')
        pilot.note(f'epoch {epoch+1}: loss {item["train_loss"]:.3f}; held-out flow {metrics["flow_mse"]:.3f}/{metrics["zero_flow_mse"]:.3f}; reverse gap {metrics["forward_vs_reverse_gap"]:.3f}')
    pilot.write_json(OUT/'train-log.json',dict(train_clips=len(tr),validation_clips=len(va),held_out_films=sorted(HELD),initial=initial,log=log,selection='minimum validation flow MSE',limitations='Proxy diagnostics, not camera-angle/style accuracy. Validation films used for checkpoint selection; no independent test set.'))


def search(args):
    clips=json.loads((OUT/'clips.json').read_text()); data=np.load(OUT/'features.npz')
    x=torch.from_numpy(data['sequences']).float()
    flow=torch.from_numpy(data['flow']).float()
    ckpt=torch.load(OUT/'head.pt',map_location='cpu',weights_only=True)
    model=TemporalHead(); model.load_state_dict(ckpt['state_dict']); model.eval()
    with torch.inference_mode(): z=torch.cat([model(x[s:s+256],flow[s:s+256]) for s in range(0,len(x),256)]).numpy()
    src=np.array([i for i,c in enumerate(clips) if c['kurosawa']]); results={}; rows=[]; overlaps={}
    baseline=json.loads((BASE/'search.json').read_text())
    for stem in TARGETS:
        tgt=np.array([i for i,c in enumerate(clips) if c['movie'].endswith('/'+stem)])
        score=z[src]@z[tgt].T; selected=[]; used_s=[]; used_t=[]; counts={}
        for flat in np.argsort(score,axis=None)[::-1]:
            a,b=divmod(int(flat),len(tgt)); s,t=int(src[a]),int(tgt[b]); sr,tr=clips[s],clips[t]
            if counts.get(sr['movie'],0)>=2: continue
            if any(clips[i]['movie']==sr['movie'] and abs(clips[i]['shot']-sr['shot'])<3 for i in used_s):continue
            if any(abs(clips[i]['shot']-tr['shot'])<3 for i in used_t):continue
            rank=len(selected)+1; name=f'{stem}-{rank}.gif'; gif_pair(OUT,name,sr,tr)
            selected.append(dict(source=s,target=t,score=float(score[a,b]),gif=name))
            used_s.append(s);used_t.append(t);counts[sr['movie']]=counts.get(sr['movie'],0)+1
            old=baseline[stem][rank-1]
            rows.append(f'<article><h2>{html.escape(stem)} #{rank}</h2><p>Trained temporal model · cosine {score[a,b]:.3f}</p><img loading="lazy" src="{name}"><details><summary>Previous heuristic baseline, same rank</summary><img loading="lazy" src="../sequence-v1/{old["gif"]}"></details></article>')
            if len(selected)==6:break
        results[stem]=selected
        overlaps[stem]=len({(p['source'],p['target']) for p in selected}&{(p['source'],p['target']) for p in baseline[stem]})
    pilot.write_json(OUT/'search.json',results)
    pilot.write_json(OUT/'report.json',dict(checkpoint_epoch=ckpt['epoch'],clips=len(clips),pairs=sum(map(len,results.values())),overlap_with_baseline=overlaps,trained_parameters=sum(p.numel() for p in model.parameters()),held_out_films=sorted(HELD),limitations=['Self-supervised visual-plus-flow temporal representation; no angle or auteur labels','Motion targets are approximate optical flow at 2fps','No human assessment of filmmaking similarity','Held-out films are validation, not independent test films', 'Holdout applies to temporal training only; frozen composition head previously saw catalogue films']))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Trained temporal model</title><style>body{background:#171b22;color:#eee;font:16px sans-serif;max-width:850px;margin:30px auto}img{width:100%}article{margin:40px 0}</style><h1>Trained temporal model</h1><p>Ordered visual features plus measured coarse optical flow. Trained to recover motion and composition changes from perturbed features. Retrieval uses the learned joint embedding. Camera angle and auteur similarity are not yet validated. Star Wars, West and Harakiri were held out from temporal-head updates, but used for checkpoint selection. The frozen composition head was previously trained across catalogue films.</p>'+''.join(rows))
    pilot.note(f'Trained-model gallery: {OUT}/index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=['extract','train','search','all'])
    p.add_argument('--threads',type=int,default=8);p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch',type=int,default=128)
    args=p.parse_args();torch.set_num_threads(args.threads)
    for name,fn in [('extract',extract),('train',train),('search',search)]:
        if args.step in (name,'all'):fn(args)

if __name__=='__main__':main()
