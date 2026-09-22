"""CPU-only: create the shared case split, fixed suites and loss weights once."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import json
from pathlib import Path

from experiments.controlled.data import (SCHEMA, ControlledDataset, alpha_from_counts,
                                         digest, evaluation_records, split_files)
import numpy as np


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def pipeline_fingerprint():
    paths = [Path(__file__).with_name('data.py'), *sorted(Path('dataset').glob('*.py'))]
    return digest({str(p.name): file_hash(p) for p in paths})


def validate_manifest(manifest):
    expected = manifest['id']
    content = {k: v for k, v in manifest.items() if k != 'id'}
    if digest(content) != expected or manifest['schema'] != SCHEMA:
        raise ValueError('Manifest checksum/schema mismatch')
    if manifest['pipeline_sha256'] != pipeline_fingerprint():
        raise ValueError('Data pipeline differs from the frozen suite. Prepare a new version.')
    from experiments.controlled.data import case_id
    sets = [{case_id(f) for f in manifest['splits'][s]} for s in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Case overlap between splits')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/3dteethseg/processed_w5')
    p.add_argument('--out', default='experiments/controlled/manifests/v1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha-samples', type=int, default=2048)
    p.add_argument('--verify', action='store_true', help='Verify every cache hash against an existing manifest')
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    if a.verify:
        manifest = json.loads((out / 'manifest.json').read_text())
        validate_manifest(manifest)
        for f, h in manifest['cache_sha256'].items():
            if file_hash(root / f) != h:
                raise ValueError(f'Cache mismatch: {f}')
        print('All cache files match the shared manifest.')
        return
    if out.exists():
        raise FileExistsError(f'{out} exists; use a new directory or --verify. Never overwrite a suite.')
    files = sorted(f.name for f in root.glob('data_*_*.pt'))
    splits = split_files(files, a.seed)
    if a.alpha_samples < 1:
        raise ValueError('alpha-samples must be positive')
    print('Hashing cache contents and preparing case-independent splits...', flush=True)
    hashes = {f: file_hash(root / f) for f in files}
    dataset = ControlledDataset(root, splits['train'], seed=a.seed, length=a.alpha_samples)
    counts = np.zeros(17, dtype=np.int64)
    patch_counts = np.zeros(17, dtype=np.int64)
    for i in range(a.alpha_samples):
        sample, _, _ = dataset[i]
        hist = np.bincount(sample[2].numpy(), minlength=17)
        counts += hist
        patch_counts += hist > 0
        if (i + 1) % 100 == 0:
            print(f'Frequency sample {i+1}/{a.alpha_samples}', flush=True)
    # Fail rather than inventing huge weights for an unobserved rare class.
    inverse = alpha_from_counts(counts, 1.).tolist()
    softened = alpha_from_counts(counts, .5).tolist()
    manifest = dict(schema=SCHEMA, seed=a.seed, splits=splits, cache_sha256=hashes,
                    pipeline_sha256=pipeline_fingerprint(),
                    cache_sizes={f: (root / f).stat().st_size for f in files},
                    alpha_samples=a.alpha_samples, counts=counts.tolist(),
                    patch_presence_counts=patch_counts.tolist(),
                    alpha={'inverse': inverse, 'sqrt': softened},
                    val=evaluation_records(splits['val'], a.seed + 1000),
                    test=evaluation_records(splits['test'], a.seed + 2000))
    manifest['id'] = digest(manifest)
    validate_manifest(manifest)
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {out}/manifest.json ({manifest["id"]}). Splits: '
          f'{ {k: len(v) for k, v in splits.items()} }')
    print('Copy this same manifest to the server; do not independently regenerate it there.')


if __name__ == '__main__':
    main()
