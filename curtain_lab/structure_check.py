import json, numpy as np
from PIL import Image, ImageFilter
recs=json.load(open('results/composition-v1/search-records.json'))
grid=np.load('results/composition-v1/search-grid.npy', mmap_mode='r')
def key(r): return (r['title'], r['shot'])
idx={key(r):i for i,r in enumerate(recs)}
junk=[('Rashomon',148),('Ran',709),('Kagemusha',491),('Ran',326),('Star Wars',41),('Star Wars',999),('Once Upon a Time in the West',868),('Yojimbo',384)]
good=[('Kagemusha',461),('Star Wars',1438),('Ran',619),('Once Upon a Time in the West',548),('Kagemusha',842),('Once Upon a Time in the West',191),('Yojimbo',189),('Once Upon a Time in the West',1084),('Star Wars',915),('High and Low',222),('Star Wars',1809),('Once Upon a Time in the West',552)]
# measure A: cell uniformity (mean pairwise cosine of 16 cell features)
def uniformity(i):
    c=grid[i,0].astype(np.float32); c/=np.linalg.norm(c,axis=1,keepdims=True)
    s=c@c.T; return (s.sum()-16)/(16*15)
# measure B: coarse edges — blur away fine texture, then gradient energy on 64px-wide luminance
def coarse(i):
    with Image.open(recs[i]['path']) as im: g=im.convert('L')
    g=g.resize((96, max(1,int(96*g.height/g.width)))).filter(ImageFilter.GaussianBlur(1.5))
    a=np.asarray(g,np.float32); a=(a-a.mean())/(a.std()+1e-3)
    gx=np.abs(np.diff(a,axis=1)).mean(); gy=np.abs(np.diff(a,axis=0)).mean()
    return gx+gy
rng=np.random.default_rng(0); sample=rng.choice(len(recs),3000,replace=False)
U=np.array([uniformity(i) for i in sample]); C=np.array([coarse(i) for i in sample])
def pct(arr,v): return (arr<v).mean()*100
for name,lst in (('JUNK',junk),('GOOD',good)):
    print(name)
    for k in lst:
        i=idx[k]; u,c=uniformity(i),coarse(i)
        print(f"  {k[0][:14]:14} {k[1]:5}  uniformity {u:.3f} (pct {pct(U,u):3.0f})  coarse-edge {c:.3f} (pct {pct(C,c):3.0f})")
print('uniformity pctiles 50/80/90/95', np.percentile(U,[50,80,90,95]).round(3))
