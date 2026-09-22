from os.path import join

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_generator import grow_patch_full_sweep, load_cached_arch, pick_incisor_seed_label
from dataset.patch_generator_midline import CENTRAL_INCISOR_LABELS, grow_patch_midline_seed

# New, additive-only file - dataset/patch_dataset.py and dataset/patch_generator.py are both
# untouched. Wires dataset/patch_generator_midline.py's midline-seeded curriculum into
# PatchTeeth3DSDataset as a SECOND sampling mode, chosen probabilistically per __getitem__ call
# alongside the existing scan-sweep mode - same structure and rationale as
# dataset/patch_dataset_whole_tooth.py's PatchTeeth3DSDatasetWithWholeTooth (see that file for why
# __getitem__ has to be overridden wholesale: there's no hook to swap just the growth function
# without editing the base class, which the project's standing rule doesn't allow).
#
# NOT combined with PatchTeeth3DSDatasetWithWholeTooth in this first pass - each new curriculum
# axis gets tested in isolation before any combining, same methodology as the whole-tooth/
# dilation_ks/color-style ablation matrix. train_patch_network_color.py guards against passing
# both --whole_tooth_patch_prob > 0 and --midline_seed_prob > 0 in the same run for this reason.
#
# Falls back to the normal scan-sweep branch whenever central incisors 8/9 aren't BOTH present in
# this arch (grow_patch_midline_seed needs both) - rare, but real arches do sometimes lack one.


class PatchTeeth3DSDatasetWithMidlineSeed(PatchTeeth3DSDataset):
    def __init__(self, *args, midline_seed_prob: float = 0.3, midline_seed_min_mm: float = 8.0,
                 midline_seed_max_mm: float = 22.0, midline_seed_stages: int = 8, **kwargs):
        """midline_seed_prob=0.0 always takes the scan-sweep branch (the rng.random() < 0.0 check
        is never true) - this is the mixing knob, not a mode switch a caller needs to separately
        enable/disable. Same NOT-byte-identical-even-at-0.0 caveat as
        PatchTeeth3DSDatasetWithWholeTooth (one extra rng.random() draw per __getitem__ call
        shifts every later draw's exact value vs. a plain PatchTeeth3DSDataset instance) - same
        distribution, different specific numbers."""
        super().__init__(*args, **kwargs)
        self.midline_seed_prob = midline_seed_prob
        self.midline_seed_min_mm = midline_seed_min_mm
        self.midline_seed_max_mm = midline_seed_max_mm
        self.midline_seed_stages = midline_seed_stages

    def __getitem__(self, index):
        pt_path = join(self.root, self.processed_folder, self.file_names[index])
        mesh, labels = load_cached_arch(pt_path)
        rng = self._rng_for(index)

        present = set(labels.tolist())
        use_midline = (rng.random() < self.midline_seed_prob
                        and CENTRAL_INCISOR_LABELS[0] in present and CENTRAL_INCISOR_LABELS[1] in present)

        if use_midline:
            min_occ = self._sample_min_occlusality(rng)
            masks = grow_patch_midline_seed(mesh, labels, min_size_mm=self.midline_seed_min_mm,
                                             max_size_mm=self.midline_seed_max_mm,
                                             n_stages=self.midline_seed_stages,
                                             slab_mm=self.slab_mm, min_occlusality=min_occ)
        else:
            seed_label = pick_incisor_seed_label(labels, rng)
            min_occ = self._sample_min_occlusality(rng)
            masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, self.width_mm,
                                                   self.height_mm, self.slab_mm, min_occ)

        stage_idx = self._sample_stage_index(len(masks), rng)
        augment_rng = rng if (self.is_train and self.augment is not None) else None
        return self._stage_tensors(mesh, labels, masks[stage_idx], augment_rng)
