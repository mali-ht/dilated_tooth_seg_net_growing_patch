import numpy as np
from scipy.sparse import csr_matrix

# below this many faces, dropout-style augmentation (holes/erosion/density subsample) is skipped
# entirely - on a patch that's already tiny, further removal risks emptying it out or destroying
# the little signal it has; noise (position/normal jitter) is still applied regardless of size,
# since it never removes faces
MIN_FACES_FOR_DROPOUT = 20


def _patch_adjacency(mesh, face_idx):
    """CSR face-adjacency matrix restricted to face_idx, indexed locally (0..len(face_idx)-1) -
    same construction as patch_generator.rectangular_footprint's connected-components step, just
    scoped to an already-selected patch instead of the whole mesh."""
    local_id = -np.ones(len(mesh.faces), dtype=np.int64)
    local_id[face_idx] = np.arange(len(face_idx))
    a, b = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    keep = (local_id[a] >= 0) & (local_id[b] >= 0)
    la, lb = local_id[a[keep]], local_id[b[keep]]
    n = len(face_idx)
    return csr_matrix((np.ones(len(la) * 2), (np.concatenate([la, lb]), np.concatenate([lb, la]))), shape=(n, n))


def punch_holes(mesh, mask, rng, max_holes=2, hole_size_range=(3, 40), max_removed_fraction=0.15):
    """
    Remove 0-max_holes small connected blobs of faces from the patch (grown by face-adjacency
    BFS from a random patch face, not by geometric radius, so it stays scale-appropriate on both
    tiny and huge patches) - simulates real scanner dropout from specular highlights, saliva,
    blood, or debris leaving small holes in an otherwise contiguous capture. Total removal is
    capped at max_removed_fraction of the patch so a run of bad luck can't gut it.
    """
    face_idx = np.where(mask)[0]
    n = len(face_idx)
    if n < MIN_FACES_FOR_DROPOUT:
        return mask

    adj = _patch_adjacency(mesh, face_idx)
    removed_local = np.zeros(n, dtype=bool)
    budget = max(1, int(n * max_removed_fraction))
    n_holes = int(rng.integers(0, max_holes + 1))

    for _ in range(n_holes):
        remaining_budget = budget - int(removed_local.sum())
        if remaining_budget <= 0:
            break
        size = min(int(rng.integers(hole_size_range[0], hole_size_range[1] + 1)), remaining_budget)
        seed = int(rng.integers(0, n))
        visited = {seed}
        frontier = [seed]
        collected = [seed]
        while frontier and len(collected) < size:
            cur = frontier.pop()
            neighbors = adj[[cur]].indices
            order = rng.permutation(len(neighbors))
            for k in order:
                nb = int(neighbors[k])
                if nb not in visited:
                    visited.add(nb)
                    frontier.append(nb)
                    collected.append(nb)
                    if len(collected) >= size:
                        break
        removed_local[collected] = True

    new_mask = mask.copy()
    new_mask[face_idx[removed_local]] = False
    return new_mask


def erode_boundary(mesh, mask, rng, max_drop_prob=0.3):
    """
    Randomly drop a fraction of the patch's own boundary faces (patch faces with fewer than 3
    patch-internal edge-neighbors, i.e. at least one edge borders a face outside the patch) -
    simulates a real scanner's ragged, imperfect capture edge instead of our generator's clean
    rectangular cutoff. Interior faces (all 3 neighbors present) are never touched.
    """
    face_idx = np.where(mask)[0]
    n = len(face_idx)
    if n < MIN_FACES_FOR_DROPOUT:
        return mask

    adj = _patch_adjacency(mesh, face_idx)
    degree = np.asarray((adj > 0).sum(axis=1)).ravel()
    is_boundary = degree < 3
    drop_prob = rng.uniform(0.0, max_drop_prob)
    drop = is_boundary & (rng.random(n) < drop_prob)

    new_mask = mask.copy()
    new_mask[face_idx[drop]] = False
    return new_mask


def density_subsample(mask, rng, keep_fraction_range=(0.7, 1.0)):
    """
    Uniformly random-thin the patch to a random keep_fraction - simulates locally sparser real
    scan density (faster sweep motion, distance-from-sensor falloff) independent of patch shape,
    unlike punch_holes (localized) or erode_boundary (edge-only).
    """
    face_idx = np.where(mask)[0]
    n = len(face_idx)
    if n < MIN_FACES_FOR_DROPOUT:
        return mask

    keep_fraction = rng.uniform(*keep_fraction_range)
    if keep_fraction >= 0.999:
        return mask
    keep = rng.random(n) < keep_fraction
    new_mask = np.zeros_like(mask)
    new_mask[face_idx[keep]] = True
    return new_mask


def jitter_positions(vertices, patch_faces, rng, sigma_mm_range=(0.0, 0.15)):
    """
    Additive per-VERTEX (not per face-corner) Gaussian position noise, restricted to vertices
    actually used by this patch - simulates scanner depth noise. Per-vertex (rather than
    independently per face-corner) matters: faces sharing a vertex must get the identical jitter
    for that vertex, or the patch would visibly tear apart at every shared edge, which no real
    scan noise looks like.
    """
    sigma = rng.uniform(*sigma_mm_range)
    if sigma <= 0:
        return vertices
    vert_idx = np.unique(patch_faces)
    noisy = vertices.copy()
    noisy[vert_idx] = noisy[vert_idx] + rng.normal(0.0, sigma, size=(len(vert_idx), 3))
    return noisy


def jitter_normals(vertex_normals, face_normals, patch_faces, face_idx, rng, sigma_range=(0.0, 0.08)):
    """
    Small additive Gaussian noise on unit normals, renormalized after - simulates normal
    estimation noise from real scan reconstruction (as opposed to the exact, noise-free normals
    computed from clean Teeth3DS ground-truth geometry). sigma=0.08 on a unit vector is roughly a
    few degrees of angular perturbation. Vertex normals are perturbed per-vertex (same sharing
    argument as jitter_positions); face normals per-face, since they aren't shared.
    """
    sigma = rng.uniform(*sigma_range)
    if sigma <= 0:
        return vertex_normals, face_normals

    vert_idx = np.unique(patch_faces)
    vn = vertex_normals.copy()
    vn[vert_idx] = vn[vert_idx] + rng.normal(0.0, sigma, size=(len(vert_idx), 3))
    vn[vert_idx] /= np.linalg.norm(vn[vert_idx], axis=1, keepdims=True) + 1e-9

    fn = face_normals.copy()
    fn[face_idx] = fn[face_idx] + rng.normal(0.0, sigma, size=(len(face_idx), 3))
    fn[face_idx] /= np.linalg.norm(fn[face_idx], axis=1, keepdims=True) + 1e-9

    return vn, fn


class SimToRealAugment:
    """
    Sim-to-real augmentation (TRAINING_CONCERNS.md #5): training exclusively on clean Teeth3DS
    ground truth risks fragility to real scan artifacts the live scanner will actually produce.
    Composes 5 independent perturbations, each with its own randomized magnitude per call:
      - punch_holes / erode_boundary / density_subsample: face DROPOUT (mask shrinks, never grows
        - we only ever simulate missing/omitted real surface, never fabricate new geometry)
      - jitter_positions / jitter_normals: geometry NOISE (same face count, perturbed coordinates)

    Train-mode only by design (see PatchTeeth3DSDataset) - never applied to the fixed validation
    set or to get_full_sequence's clean visualization/debug output.
    """

    def __init__(self, max_holes=2, hole_size_range=(3, 40), max_removed_fraction=0.15,
                 max_boundary_drop_prob=0.3, density_keep_fraction_range=(0.7, 1.0),
                 position_sigma_mm_range=(0.0, 0.15), normal_sigma_range=(0.0, 0.08)):
        self.max_holes = max_holes
        self.hole_size_range = hole_size_range
        self.max_removed_fraction = max_removed_fraction
        self.max_boundary_drop_prob = max_boundary_drop_prob
        self.density_keep_fraction_range = density_keep_fraction_range
        self.position_sigma_mm_range = position_sigma_mm_range
        self.normal_sigma_range = normal_sigma_range

    def augment_mask(self, mesh, mask, rng):
        mask = punch_holes(mesh, mask, rng, self.max_holes, self.hole_size_range, self.max_removed_fraction)
        mask = erode_boundary(mesh, mask, rng, self.max_boundary_drop_prob)
        mask = density_subsample(mask, rng, self.density_keep_fraction_range)
        return mask

    def augment_geometry(self, vertices, vertex_normals, face_normals, patch_faces, face_idx, rng):
        vertices = jitter_positions(vertices, patch_faces, rng, self.position_sigma_mm_range)
        vertex_normals, face_normals = jitter_normals(vertex_normals, face_normals, patch_faces, face_idx, rng,
                                                        self.normal_sigma_range)
        return vertices, vertex_normals, face_normals
