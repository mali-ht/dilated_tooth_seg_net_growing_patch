import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.patch_augmentation import SimToRealAugment
from dataset.patch_generator import grow_patch_full_sweep, load_cached_arch, pick_incisor_seed_label
from dataset.patch_preprocessing import PatchPreTransform
from utils.teeth_numbering import label_to_coarse_label


def collate_single_sequence(batch):
    """Use with DataLoader(dataset, batch_size=1, collate_fn=collate_single_sequence) if you're
    iterating get_full_sequence-style output through a DataLoader: unwraps the outer size-1 batch
    DataLoader would otherwise wrap around it (torch's default collate would instead try to stack
    the stages together, which fails since every stage has a different face count). NOT needed for
    normal __getitem__ use - see TRAINING_CONCERNS.md #2 for why __getitem__ went back to
    returning one randomized single patch instead of a whole arch's sequence as "one batch"."""
    return batch[0]


# sentinel distinguishing "transform not specified, use the PatchPreTransform default" from an
# explicit transform=None (raw 5-tuples, no normalization) - plain None can't do both jobs
_DEFAULT_TRANSFORM = object()


class PatchTeeth3DSDataset(Dataset):
    """
    On-the-fly occlusal/facial/lingual scan-sweep patches (dataset/patch_generator.py) drawn from
    a Teeth3DSDataset cache - see claude_code_prompt.md / MAA_ToDo.md for the design. This is a
    NEW, separate dataset: dataset/mesh_dataset.py's Teeth3DSDataset (the original full-arch
    fixed-count path) and everything in models/ are untouched - this is an additive path, not a
    replacement, and does not do its own preprocessing (point it at an existing processed_w5-style
    cache - the UNCAPPED, true-density one, built by preprocess_data.py - not the capped16k one,
    so patches carry the same density the live scanner would produce).

    __getitem__(index) returns ONE RANDOM PATCH from arch `index`'s occlusal->facial->lingual
    sweep, biased toward small/early stages via early_bias_power (see _sample_stage_index) -
    NOT the whole sequence. `index` selects the arch; which stage/size gets drawn is re-randomized
    every call in train mode, so with DataLoader(..., shuffle=True) consecutive training steps
    naturally interleave different patch sizes from different arches. See TRAINING_CONCERNS.md #2:
    an earlier version returned a whole arch's ~40-stage sequence as "one batch" (meant to be fed
    as consecutive correlated forward/backward passes) - reverted, since that's about as far from
    i.i.d. SGD steps as you can get and risks getting stuck in local minima from the correlation.

    Each item is transform(mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals,
    labels) PLUS the patch's real mm^2 area appended as a final element - by default transform is
    PatchPreTransform (dataset/patch_preprocessing.py, Component 3's patch-invariant
    normalization), so items are (pos, x, labels, area_mm2) 4-tuples; pass transform=None for the
    raw untransformed 5-tuple + area_mm2 (a 6-tuple) instead (e.g. for visualization/debug use,
    matching process_mesh()'s format in dataset/mesh_dataset.py). area_mm2 is Component 4's
    (models/patch_dilated_tooth_seg_network.py) area-gated dilation input - see
    models/patch_collate.py, which expects exactly the (pos, x, labels, area_mm2) shape and bundles
    it with precomputed neighbor indices ready for the model's forward().

    In train mode, the seed tooth, sweep side/end choices, which stage gets sampled, the
    occlusal-vs-wall threshold (min_occlusality, drawn from min_occlusality_range - see
    TRAINING_CONCERNS.md #4), AND the sim-to-real augmentation (holes/erosion/density/noise - see
    dataset/patch_augmentation.py, TRAINING_CONCERNS.md #5) all use fresh randomness every call (a
    different, differently-perturbed patch each epoch). In eval mode, everything is seeded
    per-index (val_seed + index) and augmentation is OFF entirely, so the same clean patch is
    reproduced every epoch - a fixed, stable validation set as specified in claude_code_prompt.md.

    The full per-arch sequence this samples from is still available via get_full_sequence(index)
    (e.g. for visualization/debugging) - it just isn't what __getitem__ returns by default anymore.
    get_full_sequence always uses the fixed min_occlusality value (not the randomized range) and
    never augments, so it gives one consistent, inspectable curriculum rather than a different
    threshold/perturbation per stage.
    """

    def __init__(self, root: str, processed_folder: str = 'processed_w5', num_classes: int = 17,
                 is_train: bool = True, train_test_split: int = 1, val_seed: int = 0,
                 width_mm: float = 14.0, height_mm: float = 14.0, slab_mm: float = 16.0,
                 min_occlusality: float = 0.6, min_occlusality_range=(0.2, 0.8),
                 early_bias_power: float = 2.5, augment=True, transform=_DEFAULT_TRANSFORM):
        if num_classes not in (5, 17):
            raise ValueError(f'num_classes should be 5 or 17, not {num_classes}')
        self.root = root
        self.processed_folder = processed_folder
        self.num_classes = num_classes
        self.is_train = is_train
        self.train_test_split = train_test_split
        self.val_seed = val_seed
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.slab_mm = slab_mm
        self.min_occlusality = min_occlusality
        # None disables randomization (every __getitem__ call uses the fixed min_occlusality
        # above); otherwise a (low, high) range that __getitem__ draws uniformly from each call -
        # domain randomization over how "top-down vs. tilted" the simulated scan angle is, so the
        # model doesn't overfit to one fixed occlusal/wall ratio. See TRAINING_CONCERNS.md #4 for
        # the measurements this range was picked from.
        self.min_occlusality_range = min_occlusality_range
        self.early_bias_power = early_bias_power
        # True -> SimToRealAugment() with its default magnitudes; an explicit SimToRealAugment
        # instance -> used as-is (custom magnitudes); False/None -> disabled (e.g. for eval, or
        # to isolate augmentation's effect while debugging). Only ever applied in train mode
        # regardless of this setting - see __getitem__.
        if isinstance(augment, SimToRealAugment):
            self.augment = augment
        elif augment:
            self.augment = SimToRealAugment()
        else:
            self.augment = None
        # not specified at all -> PatchPreTransform (Component 3) by default; explicit
        # transform=None -> raw 5-tuples; anything else -> used as-is
        self.transform = PatchPreTransform() if transform is _DEFAULT_TRANSFORM else transform

        self.file_names = []
        self._set_file_index(is_train)

    def _set_file_index(self, is_train: bool):
        if self.train_test_split == 1:
            split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else \
                ['testing_lower.txt', 'testing_upper.txt']
        elif self.train_test_split == 2:
            split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
                else ['private-testing-set.txt']
        else:
            raise ValueError(f'train_test_split should be 1 or 2, not {self.train_test_split}')
        for f in split_files:
            with open(f'data/3dteethseg/raw/{f}') as file:
                for l in file:
                    l = f'data_{l.rstrip()}.pt'
                    if os.path.isfile(join(self.root, self.processed_folder, l)):
                        self.file_names.append(l)

    def __len__(self):
        return len(self.file_names)

    def _sample_stage_index(self, n_stages, rng):
        """Biased toward small index (early/small patches): u ~ Uniform(0,1) raised to a power >
        1 concentrates mass near 0. early_bias_power=1 would be uniform over all stages."""
        return int((rng.random() ** self.early_bias_power) * (n_stages - 1))

    def _stage_tensors(self, mesh, labels, mask, augment_rng=None):
        if augment_rng is not None:
            mask = self.augment.augment_mask(mesh, mask, augment_rng)

        face_idx = np.where(mask)[0]
        # real mm^2 area, BEFORE PatchPreTransform's fixed-SCALE_MM rescale discards absolute
        # scale - Component 4's area-gated dilation (models/patch_dilated_tooth_seg_network.py)
        # needs the true area, not a normalized one. Computed on the un-augmented face set
        # deliberately: augmentation (holes/erosion/density subsample) simulates scanner
        # imperfection on top of a patch the robot's own sweep logic already decided to capture
        # at this real-world extent - area-gating should reflect what was actually swept, not an
        # artifact of how much dropout this particular draw happened to apply.
        area_mm2 = float(mesh.area_faces[face_idx].sum())

        patch_faces = mesh.faces[face_idx]
        vertices, vertex_normals, face_normals = mesh.vertices, mesh.vertex_normals, mesh.face_normals
        if augment_rng is not None:
            vertices, vertex_normals, face_normals = self.augment.augment_geometry(
                vertices, vertex_normals, face_normals, patch_faces, face_idx, augment_rng)

        mesh_faces = torch.from_numpy(patch_faces.copy()).float()
        mesh_triangles = torch.from_numpy(vertices[patch_faces].copy()).float()
        mesh_face_normals = torch.from_numpy(face_normals[face_idx].copy()).float()
        mesh_vertices_normals = torch.from_numpy(vertex_normals[patch_faces].copy()).float()

        patch_labels = labels[face_idx]
        if self.num_classes == 5:
            patch_labels = label_to_coarse_label(patch_labels)
        patch_labels = torch.from_numpy(patch_labels).long()

        raw = (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, patch_labels)
        transformed = self.transform(raw) if self.transform is not None else raw
        return (*transformed, area_mm2)

    def _rng_for(self, index):
        return np.random.default_rng() if self.is_train else np.random.default_rng(self.val_seed + index)

    def _sample_min_occlusality(self, rng):
        if self.min_occlusality_range is None:
            return self.min_occlusality
        low, high = self.min_occlusality_range
        return float(rng.uniform(low, high))

    def __getitem__(self, index):
        pt_path = join(self.root, self.processed_folder, self.file_names[index])
        mesh, labels = load_cached_arch(pt_path)
        rng = self._rng_for(index)

        seed_label = pick_incisor_seed_label(labels, rng)
        min_occ = self._sample_min_occlusality(rng)
        masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, self.width_mm, self.height_mm,
                                               self.slab_mm, min_occ)

        stage_idx = self._sample_stage_index(len(masks), rng)
        augment_rng = rng if (self.is_train and self.augment is not None) else None
        return self._stage_tensors(mesh, labels, masks[stage_idx], augment_rng)

    def get_full_sequence(self, index):
        """
        The full ordered occlusal -> facial -> lingual sweep for arch `index`, as a (sequence,
        stages) pair - NOT what __getitem__ returns (see class docstring / TRAINING_CONCERNS.md
        #2). For visualization/debugging, or any future use that actually wants the whole
        progression rather than one randomly-sampled patch.
        """
        pt_path = join(self.root, self.processed_folder, self.file_names[index])
        mesh, labels = load_cached_arch(pt_path)
        rng = self._rng_for(index)

        seed_label = pick_incisor_seed_label(labels, rng)
        masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, self.width_mm, self.height_mm,
                                               self.slab_mm, self.min_occlusality)

        sequence = [self._stage_tensors(mesh, labels, mask) for mask in masks]
        return sequence, stages
