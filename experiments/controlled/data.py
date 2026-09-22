"""Versioned experiment data path; historical training stays unchanged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.patch_augmentation import SimToRealAugment
from dataset.patch_generator import (grow_patch_full_sweep, load_cached_arch,
                                     pick_incisor_seed_label)
from dataset.patch_generator_midline import grow_patch_midline_seed
from dataset.patch_preprocessing import PatchPreTransform
from dataset.patch_preprocessing_color import PatchPreTransformWithColor
from dataset.patch_preprocessing_realistic_color_hsv import PatchPreTransformWithRealisticColorHSV
from dataset.patch_color_augmentation import NO_COLOR_SENTINEL
from models.patch_collate import PatchCollator

PAIRS = ((4, 5), (12, 13), (1, 2), (2, 3), (14, 15), (15, 16))
SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def case_id(filename):
    stem = Path(filename).stem.removeprefix('data_')
    case, arch = stem.rsplit('_', 1)
    if arch not in ('upper', 'lower'):
        raise ValueError(f'Unrecognized arch filename: {filename}')
    return case


def split_files(files, seed=42):
    """Keep paired jaws together; filename case IDs must match dataset provenance."""
    files = sorted(files)
    cases = sorted({case_id(f) for f in files})
    if len(cases) < 10:
        raise ValueError('At least 10 cases are required for the 70/15/15 split')
    np.random.default_rng(seed).shuffle(cases)
    ntrain, nval = int(.7 * len(cases)), int(.15 * len(cases))
    groups = {'train': set(cases[:ntrain]), 'val': set(cases[ntrain:ntrain+nval]),
              'test': set(cases[ntrain+nval:])}
    return {k: [f for f in files if case_id(f) in ids] for k, ids in groups.items()}


def stage_index(n, power, rng):
    if n < 1 or power <= 0:
        raise ValueError('Positive stage count and power required')
    return min(n - 1, int(rng.random() ** power * n))


def alpha_from_counts(counts, exponent):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (17,) or not np.all(counts > 0):
        raise ValueError('Every class needs coverage in the fixed training-frequency sample')
    weights = counts ** -exponent
    return torch.tensor(weights * 17 / weights.sum(), dtype=torch.float32)


def stable_rng(*parts):
    seed = int(digest(parts)[:16], 16)
    return np.random.default_rng(seed)


class EpochSampler(Sampler):
    """Lightning calls set_epoch before iteration, including resumed epochs.

    Epoch travels in each key; persistent workers never rely on mutable epoch state.
    """
    def __init__(self, length, start_epoch=0):
        self.length, self.epoch = length, start_epoch
        self.start_epoch = start_epoch

    def set_epoch(self, epoch):
        # Lightning can prefetch using the saved epoch before advancing its
        # completed-epoch progress during resume. Never replay that epoch's keys.
        self.epoch = max(epoch, self.start_epoch)

    def __iter__(self):
        return iter((self.epoch, i) for i in range(self.length))

    def __len__(self):
        return self.length


def select_mask(mesh, labels, rng, mode, power=1.5):
    """Use real footprints, never label-clipped teeth as model input boundaries."""
    present = set(np.unique(labels).tolist())
    pair = None
    actual_mode = mode
    if mode == 'pair':
        eligible = [p for p in PAIRS if set(p) <= present]
        if eligible:
            pair = eligible[int(rng.integers(len(eligible)))]
        else:
            actual_mode = 'sweep'
    if mode == 'midline':
        if {8, 9} <= present:
            pair = (8, 9)
        else:
            actual_mode = 'sweep'
    if pair is not None:
        masks = grow_patch_midline_seed(mesh, labels, min_size_mm=8, max_size_mm=22,
                                       n_stages=8, min_occlusality=float(rng.uniform(.2, .8)),
                                       tooth_a=pair[0], tooth_b=pair[1])
    else:
        # End-of-arch starts deliberately include molar-positive and near-terminal
        # negatives. Actual visibility is recorded from the selected labels below.
        if mode == 'terminal':
            candidates = sorted(present & {1, 2, 3, 14, 15, 16})
            seed = int(rng.choice(candidates)) if candidates else pick_incisor_seed_label(labels, rng)
            if not candidates:
                actual_mode = 'sweep'
        else:
            seed = pick_incisor_seed_label(labels, rng)
        masks, _ = grow_patch_full_sweep(mesh, labels, seed, rng, 14., 14., 16.,
                                         float(rng.uniform(.2, .8)))
    index = 0 if mode == 'early' else stage_index(len(masks), power, rng)
    return masks[index], {'requested_mode': mode, 'mode': actual_mode,
                          'pair': list(pair) if pair else [], 'stage': index}


class ControlledDataset(Dataset):
    def __init__(self, root, files, *, seed=42, length=None, targeted=False,
                 records=None, max_faces=20000, color='mosaic', augment=True):
        self.root, self.files = Path(root), list(files)
        self.seed, self.length, self.targeted = seed, length or len(files), targeted
        self.records, self.max_faces = records, max_faces
        self.augmentation = SimToRealAugment() if augment else None
        self.transform = {'mosaic': PatchPreTransformWithRealisticColorHSV,
                          'flat': PatchPreTransformWithColor, 'none': PatchPreTransform}[color]()

    def __len__(self):
        return len(self.records) if self.records is not None else self.length

    def __getitem__(self, key):
        if self.records is not None:
            record = self.records[key]
            filename, mode = record['file'], record['mode']
            rng = stable_rng(record['seed'], filename, mode)
        else:
            epoch, index = key if isinstance(key, tuple) else (0, key)
            # Independent streams keep file selection paired across all four runs.
            filename = self.files[int(stable_rng(self.seed, epoch, index, 'file').integers(len(self.files)))]
            mode_draw = stable_rng(self.seed, epoch, index, 'mode').random()
            mode = ('pair' if mode_draw < .2 else 'terminal' if mode_draw < .4 else 'sweep') if self.targeted else 'sweep'
            rng = stable_rng(self.seed, epoch, index, 'patch')
            record = {'dropout': 0.0}
        mesh, labels = load_cached_arch(str(self.root / filename))
        mask, metadata = select_mask(mesh, labels, rng, mode)
        if self.augmentation is not None:
            mask = self.augmentation.augment_mask(mesh, mask, rng)
        face_idx = np.flatnonzero(mask)
        if len(face_idx) == 0:
            raise ValueError(f'Empty patch: {filename}, {metadata}')
        area = float(mesh.area_faces[face_idx].sum())
        if len(face_idx) > self.max_faces:
            face_idx = np.sort(rng.choice(face_idx, self.max_faces, replace=False))
        vertices, vn, fn = mesh.vertices, mesh.vertex_normals, mesh.face_normals
        if self.augmentation is not None:
            vertices, vn, fn = self.augmentation.augment_geometry(vertices, vn, fn,
                                                                 mesh.faces[face_idx], face_idx, rng)
        raw = (torch.tensor(mesh.faces[face_idx].copy(), dtype=torch.float32),
               torch.tensor(vertices[mesh.faces[face_idx]].copy(), dtype=torch.float32),
               torch.tensor(vn[mesh.faces[face_idx]].copy(), dtype=torch.float32),
               torch.tensor(fn[face_idx].copy(), dtype=torch.float32),
               torch.tensor(labels[face_idx].copy(), dtype=torch.long))
        pos, x, target = self.transform(raw)
        dropout = record.get('dropout', 0.)
        if dropout and x.shape[1] == 27:
            drop_rng = stable_rng(record['seed'], filename, 'color-dropout')
            drop = drop_rng.random(len(target)) < dropout
            x[drop, 24:27] = torch.tensor(NO_COLOR_SENTINEL / 127.5 - 1, dtype=x.dtype)
        # Horvitz-Thompson estimate if capped; exact original-face areas otherwise.
        # Sampling is uniform before geometry augmentation; these are physical areas.
        face_area = torch.tensor(mesh.area_faces[face_idx].copy(), dtype=torch.float32)
        face_area *= int(mask.sum()) / len(face_idx)
        metadata.update(file=filename, arch=filename.rsplit('_', 1)[-1].split('.')[0],
                        case=case_id(filename), area_mm2=area,
                        terminal_in_arch=[int(k in labels) for k in (1, 16)],
                        dropout=dropout, sample_id=record.get('id', str(key)))
        return (pos, x, target, area), metadata, face_area


def collate(batch, k=32, dilation_ks=(200, 900, 1800)):
    sample, metadata, face_area = batch[0]
    # Stable neighbors for identical geometry, independent of worker scheduling.
    h = hashlib.sha256(sample[0].numpy().tobytes()).digest()
    seed = int.from_bytes(h[:4], 'little')
    return {'patch': PatchCollator(k=k, dilation_ks=dilation_ks, seed=seed)([sample]), 'metadata': metadata,
            'face_area': face_area}


def evaluation_records(files, seed):
    records = []
    for filename in sorted(files):
        for mode in ('sweep', 'early', 'midline', 'pair', 'terminal'):
            for dropout in (0., .6):
                records.append({'id': f'{filename}:{mode}:drop{dropout}', 'file': filename,
                                'mode': mode, 'seed': seed, 'dropout': dropout})
    return records
