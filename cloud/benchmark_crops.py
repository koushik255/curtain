import json
import tarfile
import tempfile
from pathlib import Path
import modal
from cloud.benchmark_lighting import image, volume

app = modal.App('curtain-crop-benchmark')


@app.function(image=image,gpu='L4',cpu=8,memory=16384,timeout=1800,
              volumes={'/data':volume},max_containers=1)
def benchmark(external_queries):
    from benchmarks.lighting import run, CROP_CONDITIONS
    runs={'original':Path('/data/runs/l4-50k'),
          'lighting_v1':Path('/data/runs/l4-50k-lighting-v1-20260909'),
          'crop_v1':Path('/data/runs/l4-50k-crop-v1-20260909')}
    with tempfile.TemporaryDirectory(prefix='crop-benchmark-') as tmp:
        data=Path(tmp)
        with tarfile.open('/data/datasets/curtain-selected-screenshots-50k.tar') as archive:
            archive.extractall(data,filter='data')
        lawrence=data/'Lawrence_of_Arabia_1962'
        lawrence.mkdir()
        with tarfile.open('/data/datasets/l4-50k-lbfive/Lawrence_of_Arabia_1962.tar') as archive:
            archive.extractall(lawrence,filter='data')
        report=run(data,{k:p/'best.pt' for k,p in runs.items()},
                   {k:json.loads((p/'split.json').read_text()) for k,p in runs.items()},
                   Path('/data/comparisons/crop-v1-broad-20260909.json'),
                   conditions=CROP_CONDITIONS,external_queries=external_queries)
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    queries={
        'difficult_lawrence':Path('/home/koushik/.codex/attachments/fbe31928-114a-4458-8b96-910e74189ec2/codex-clipboard-0c24ed5d-c193-44aa-938c-d1257244e533.png').read_bytes(),
        'original_lawrence_control':Path('/home/koushik/.codex/attachments/b7c51f74-35fb-4aef-b783-d9f4af5a3bae/codex-clipboard-f1aa0b19-13e9-4b6d-98be-69ebc84b9051.png').read_bytes()}
    report=benchmark.remote(queries)
    output=Path('trained_models/crop-v1-broad-20260909.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'COMPLETE: {output}; {report["elapsed_seconds"]:.1f} seconds',flush=True)
