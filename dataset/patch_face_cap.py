import hashlib

import numpy as np
import torch

# New, additive-only file - dataset/patch_dataset.py, dataset/patch_preprocessing.py and
# models/patch_collate.py are all untouched. Fixes a real overnight crash (2026-08-19): both
# color training runs died with torch.AcceleratorError: CUDA error: the launch timed out and was
# terminated. journalctl/dmesg showed the actual cause: NVRM Xid 8 ("RC watchdog: GPU is probably
# locked! Notify Timeout Seconds: 7") - this machine's RTX 5090 also drives the desktop (Xorg/
# gnome-shell show up in nvidia-smi), so the driver kills any single CUDA kernel that runs longer
# than ~7s to keep the desktop from freezing.
#
# Root cause: models/layer.py's knn() (protected, unmodified) does an UNCAPPED O(N^2)
# torch.cdist whenever no precomputed neighbor index is passed - edge_graph_conv_block2/3 in
# models/patch_dilated_tooth_seg_network.py never pass one (only block1 gets local_idx from
# models/patch_collate.py). models/patch_collate.py applies no face-count cap at all during
# training (confirmed - nothing in that file bounds N). An occasional very large patch (late-
# stage/full-arch growth stages) makes that single kernel run long enough to trip the watchdog.
# Live inference already hit this exact same O(N^2) concern and fixed it with
# cap_face_count/max_model_faces=20000 (realtime/mesh_viewer_segmented.py) - this reuses
# that same default value, both to stop the crash and to keep train/inference face-count
# distributions consistent with each other.

DEFAULT_MAX_FACES = 20000


class MaxFaceCapTransform:
    """Wraps any inner PatchPreTransform-style transform (5-tuple in: mesh_faces, mesh_triangles,
    mesh_vertices_normals, mesh_face_normals, labels -> whatever the inner transform returns).
    If the patch exceeds max_faces, randomly subsamples all five parallel per-face tensors down
    to max_faces (same kept indices across all five, so geometry/normals/labels stay aligned)
    BEFORE handing off to the inner transform - then the inner transform, and everything
    downstream, never sees more than max_faces faces.

    Does NOT touch area_mm2 - that's computed by the caller (dataset/patch_dataset.py's
    _stage_tensors, untouched) from the FULL, un-capped face set before this transform ever runs,
    so area-gated dilation still sees the true physical extent of what the scanner swept, exactly
    like live inference's own cap_face_count leaves accumulated-area semantics alone too.

    The subsample seed is derived from the patch's own label content (same pattern as
    dataset/patch_preprocessing_color.py's PatchPreTransformWithColor, for the same reason: a
    shared stateful RNG advancing across calls is NOT actually reproducible across runs once
    DataLoader workers are involved, since which worker processes which index varies run to run.
    Content-derived seeding sidesteps that - the same patch gets the same subsample every time,
    which matters for PatchTeeth3DSDataset's val split, documented there as a fixed, stable
    validation set).
    """

    def __init__(self, inner, max_faces: int = DEFAULT_MAX_FACES):
        self.inner = inner
        self.max_faces = max_faces

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        n = mesh_faces.shape[0]
        if n > self.max_faces:
            seed = int(hashlib.sha256(labels.numpy().tobytes()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            keep = torch.from_numpy(np.sort(rng.choice(n, size=self.max_faces, replace=False)))
            mesh_faces = mesh_faces[keep]
            mesh_triangles = mesh_triangles[keep]
            mesh_vertices_normals = mesh_vertices_normals[keep]
            mesh_face_normals = mesh_face_normals[keep]
            labels = labels[keep]
            data = (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels)
        return self.inner(data)
