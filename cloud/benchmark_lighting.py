from pathlib import Path
import json
import tarfile
import tempfile
import modal

app = modal.App('curtain-lighting-benchmark')
volume = modal.Volume.from_name('curtain-training-data')
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install('numpy>=2.2,<3', 'pillow>=11,<13', 'torch>=2.8,<3', 'torchvision>=0.23,<1')
         .add_local_dir('curtain_ml', remote_path='/root/curtain_ml', copy=True)
         .add_local_dir('benchmarks', remote_path='/root/benchmarks', copy=True))


@app.function(image=image, gpu='L4', cpu=8, memory=16384, timeout=1800,
              volumes={'/data': volume}, max_containers=1)
def benchmark():
    from benchmarks.lighting import run
    runs = {'original': Path('/data/runs/l4-50k'),
            'lighting_v1': Path('/data/runs/l4-50k-lighting-v1-20260909')}
    with tempfile.TemporaryDirectory(prefix='lighting-benchmark-') as tmp:
        with tarfile.open('/data/datasets/curtain-selected-screenshots-50k.tar') as archive:
            archive.extractall(tmp, filter='data')
        report = run(Path(tmp), {name: path/'best.pt' for name,path in runs.items()},
                     {name: json.loads((path/'split.json').read_text()) for name,path in runs.items()},
                     Path('/data/comparisons/lighting-v1-broad-20260909.json'))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    result = benchmark.remote()
    output = Path('trained_models/lighting-v1-broad-20260909.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'COMPLETE: {output}; {result["elapsed_seconds"]:.1f} benchmark seconds', flush=True)
