from os.path import join

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_generator import grow_patch_full_sweep, load_cached_arch, pick_incisor_seed_label
from dataset.patch_generator_whole_tooth import DEFAULT_MARGIN_MM, grow_patch_whole_teeth

# New, additive-only file - dataset/patch_dataset.py and dataset/patch_generator.py are both
# untouched. Wires dataset/patch_generator_whole_tooth.py's whole-tooth curriculum into
# PatchTeeth3DSDataset as a SECOND sampling mode, chosen probabilistically per __getitem__ call
# alongside the existing scan-sweep mode - per Docs/MAA_ToDo.md's own step 4 ("NOT a replacement
# path"). Implemented as a subclass overriding __getitem__ rather than a parameter on the
# existing class, since the growth FUNCTION itself needs to change per-draw (not just the
# transform, which is why the color feature only needed transform= and never needed a subclass) -
# there's no existing hook in PatchTeeth3DSDataset.__getitem__ for swapping that in without
# editing it directly, which the project's standing rule doesn't allow.
#
# __getitem__'s body below duplicates PatchTeeth3DSDataset's own scan-sweep branch verbatim
# (there's no way to call "the rest of the parent's __getitem__" after checking one condition) -
# everything else (_rng_for, _sample_stage_index, _sample_min_occlusality, _stage_tensors) is
# reused via inheritance unchanged, so the two branches use the exact same logic the parent's own
# single-mode behavior does. NOT byte-identical output even at whole_tooth_patch_prob=0.0 though -
# the mode-decision rng.random() call is itself one extra draw against the shared per-index rng,
# which shifts every later draw's exact value vs. a plain PatchTeeth3DSDataset instance (verified
# empirically). Same distribution, different specific numbers - see __init__'s own docstring.


class PatchTeeth3DSDatasetWithWholeTooth(PatchTeeth3DSDataset):
    def __init__(self, *args, whole_tooth_patch_prob: float = 0.3,
                 whole_tooth_margin_mm: float = DEFAULT_MARGIN_MM, **kwargs):
        """whole_tooth_patch_prob=0.0 always takes the scan-sweep branch (the rng.random() < 0.0
        check is never true) - this is the mixing knob, not a mode switch a caller needs to
        separately enable/disable. NOT byte-identical to a plain PatchTeeth3DSDataset even at
        0.0, though - the mode-decision rng.random() call itself is one extra draw against the
        shared per-index rng, which shifts every subsequent draw's exact value (verified
        empirically - this was the actual first assumption, corrected once tested). What IS true
        at 0.0, and is the property that actually matters: every draw uses scan-sweep growth,
        with the same distribution/statistics as the parent class, just its own valid point in
        the rng sequence rather than an identical one."""
        super().__init__(*args, **kwargs)
        self.whole_tooth_patch_prob = whole_tooth_patch_prob
        self.whole_tooth_margin_mm = whole_tooth_margin_mm

    def __getitem__(self, index):
        pt_path = join(self.root, self.processed_folder, self.file_names[index])
        mesh, labels = load_cached_arch(pt_path)
        rng = self._rng_for(index)

        seed_label = pick_incisor_seed_label(labels, rng)

        if rng.random() < self.whole_tooth_patch_prob:
            masks, stages = grow_patch_whole_teeth(mesh, labels, seed_label, rng,
                                                     margin_mm=self.whole_tooth_margin_mm)
        else:
            min_occ = self._sample_min_occlusality(rng)
            masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, self.width_mm,
                                                   self.height_mm, self.slab_mm, min_occ)

        stage_idx = self._sample_stage_index(len(masks), rng)
        augment_rng = rng if (self.is_train and self.augment is not None) else None
        return self._stage_tensors(mesh, labels, masks[stage_idx], augment_rng)
