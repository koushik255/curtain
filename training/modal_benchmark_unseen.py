import json
from pathlib import Path
import modal

app=modal.App('curtain-unseen-benchmark')
volume=modal.Volume.from_name('curtain-training-data')
image=(modal.Image.debian_slim(python_version='3.12').uv_pip_install('numpy>=2.2,<3','pillow>=11,<13','torch>=2.8,<3','torchvision>=0.23,<1').add_local_dir('training',remote_path='/root/training',copy=True))

@app.function(image=image,gpu='L4',cpu=8,memory=16384,timeout=3600,volumes={'/data':volume},max_containers=1)
def benchmark(collection_json: str):
    from training.benchmark_unseen import main
    root=Path('/data/datasets/l4-50k-lbfive')
    collection=json.loads(collection_json)
    split=json.loads(Path('/data/runs/l4-50k/split.json').read_text())
    return main(root,collection,{'original':Path('/data/runs/l4-50k/best.pt'),'lighting_v1':Path('/data/runs/l4-50k-lighting-v1-20260909/best.pt'),'crop_v1':Path('/data/runs/l4-50k-crop-v1-20260909/best.pt')},split,'/data/comparisons/unseen-crop-v1-20260910.json')

@app.local_entrypoint()
def local():
    collection=Path('trained_indexes/l4-50k-lbfive/collection.json').read_text()
    result=benchmark.remote(collection)
    output=Path('trained_models/unseen-crop-v1-20260910.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Benchmark completed: {output}')
