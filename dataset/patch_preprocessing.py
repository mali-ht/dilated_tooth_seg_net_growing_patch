import torch

DEFAULT_SCALE_MM = 15.0


class PatchPreTransform(object):
    """
    Patch-invariant feature builder + normalization (Component 3 of claude_code_prompt.md) - a
    NEW, separate transform from dataset/preprocessing.py's PreTransform (untouched), which does
    WHOLE-MESH z-score/min-max normalization using that specific mesh's own point statistics.
    That's unsuitable for patches: the same tooth would land at a different learned scale
    depending on how much surrounding context happened to be in that particular patch (a lone
    incisor patch vs. that same incisor inside a full-arch patch have very different means/stds).

    This scheme instead:
      - centers the patch by its OWN centroid (mean face-corner position of just this patch, not
        the whole arch's)
      - divides all positions by a FIXED SCALE_MM constant (config, NOT per-patch std/min/max), so
        the same physical tooth lands at the same scale regardless of patch size
      - leaves normals raw (already unit vectors) - no z-scoring, nothing patch-dependent at all

    Produces the same 24-dim feature layout as PreTransform (3 triangle vertices + face center +
    3 vertex normals + face normal), so it drops into the existing model (feature_dim=24)
    unchanged. The point of this identical, patch-independent formula is that it is applied the
    same way at both train and inference time - unlike PreTransform's per-mesh statistics, nothing
    here depends on what else is in the patch, which is exactly what a live, growing scan needs
    (the scanner doesn't know in advance how big the final patch will be).
    """

    def __init__(self, scale_mm: float = DEFAULT_SCALE_MM, classes: int = 17):
        self.scale_mm = scale_mm
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 24).float()

        centroid = mesh_triangles.reshape(-1, 3).mean(dim=0)

        x[:, :3] = (mesh_triangles[:, 0] - centroid) / self.scale_mm
        x[:, 3:6] = (mesh_triangles[:, 1] - centroid) / self.scale_mm
        x[:, 6:9] = (mesh_triangles[:, 2] - centroid) / self.scale_mm
        x[:, 9:12] = (mesh_triangles.mean(dim=1) - centroid) / self.scale_mm
        x[:, 12:15] = mesh_vertices_normals[:, 0]
        x[:, 15:18] = mesh_vertices_normals[:, 1]
        x[:, 18:21] = mesh_vertices_normals[:, 2]
        x[:, 21:] = mesh_face_normals

        pos = x[:, 9:12]

        return pos, x, labels
