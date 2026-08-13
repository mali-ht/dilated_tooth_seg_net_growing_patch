import heapq
import pickle

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# label scheme from utils/teeth_numbering.py: 1-16 mirrored L/R by position-in-arch, same
# regardless of upper/lower arch. 8/9 = central incisors, 7/10 = lateral incisors - these four
# correspond to Universal Numbering System teeth 7,8,9,10 (upper) / 23,24,25,26 (lower), which is
# where patch seeds must start (canines, label 6/11, are excluded).
INCISOR_LABELS = {7, 8, 9, 10}


def load_cached_arch(pt_path):
    """
    Reconstruct a trimesh.Trimesh + per-face label array from a Teeth3DSDataset cache file
    (the raw process_mesh() tuple - see dataset/mesh_dataset.py). The shared vertex buffer is
    recovered from mesh_faces (original vertex indices) + mesh_triangles (per-face expanded
    coordinates), since the cache only stores the latter.
    """
    with open(pt_path, 'rb') as f:
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = pickle.load(f)
    faces = mesh_faces.numpy().astype(np.int64)
    num_vertices = int(faces.max()) + 1
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    vertices[faces.reshape(-1)] = mesh_triangles.numpy().reshape(-1, 3)
    # process=False: keep faces/vertices exactly as-is so face indices stay aligned with `labels`
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, labels.numpy()


def pick_incisor_seed_label(labels, rng):
    """Pick a random incisor tooth label (central or lateral) present in this arch as a seed."""
    present_incisors = sorted(INCISOR_LABELS & set(np.unique(labels).tolist()))
    return int(rng.choice(present_incisors))


# Labels 1-16 are already in anatomical arch order in this repo's numbering scheme (1 = one
# side's 3rd molar ... 8/9 = the two central incisors at the midline ... 16 = the other side's
# 3rd molar - see utils/teeth_numbering.py's _teeth_codes_lower/upper). Walking tooth-by-tooth
# along the arch is therefore just walking this integer range - no curve-fitting needed.
TOOTH_LABELS = list(range(1, 17))


def order_teeth_from_seed(seed_label, present_labels, rng):
    """
    Build a tooth-by-tooth visiting order starting at seed_label and sweeping ONE side of the
    arch to completion before starting the other (see TOOTH_LABELS), skipping any tooth label not
    actually present in this arch (e.g. a missing 3rd molar). Which side finishes first is random
    per call, for training diversity - the two sides are anatomically equivalent.
    """
    left = [l for l in range(seed_label - 1, 0, -1) if l in present_labels]
    right = [l for l in range(seed_label + 1, 17) if l in present_labels]
    first, second = (left, right) if rng.random() < 0.5 else (right, left)
    return [seed_label] + first + second


def order_arch_from_one_end(present_labels, rng):
    """
    A single continuous tooth order spanning the whole arch, from one back tooth (whichever end
    of the present label range - 3rd molars are often missing, so this is NOT hardcoded to
    literal labels 1/16) straight through to the other. Used for the facial/lingual wall sweeps,
    which move through the arch once rather than growing outward from a front seed. Which end
    starts is random per call, for training diversity.
    """
    ordered = sorted(present_labels)
    return ordered if rng.random() < 0.5 else list(reversed(ordered))


def tooth_local_frame(mesh, labels, tooth_label):
    """A tooth's own centroid and average (unit) face normal - the position and viewing
    direction a scanner would use if held normal to that specific tooth."""
    idx = np.where(labels == tooth_label)[0]
    origin = mesh.triangles_center[idx].mean(axis=0)
    normal = mesh.face_normals[idx].mean(axis=0)
    return origin, normal / np.linalg.norm(normal)


def tooth_tangent_direction(mesh, labels, tooth_label, present_labels, normal_axis):
    """
    Direction along the arch at this tooth's position, from its geometric neighbor(s) (label-1,
    label+1 - NOT the growth-visiting order, so this is stable regardless of where growth
    started), projected to be perpendicular to normal_axis so it lies in the local tangent plane.
    Points from the lower-label neighbor toward the higher-label one, a fixed convention along
    the whole arch. Falls back to an arbitrary in-plane direction if a tooth has no neighbors
    present at all (isolated arch fragment).
    """
    neighbors = [l for l in (tooth_label - 1, tooth_label + 1) if 1 <= l <= 16 and l in present_labels]
    if len(neighbors) == 2:
        direction = tooth_local_frame(mesh, labels, neighbors[1])[0] - tooth_local_frame(mesh, labels, neighbors[0])[0]
    elif len(neighbors) == 1:
        origin = tooth_local_frame(mesh, labels, tooth_label)[0]
        neighbor_origin = tooth_local_frame(mesh, labels, neighbors[0])[0]
        direction = (origin - neighbor_origin) if neighbors[0] < tooth_label else (neighbor_origin - origin)
    else:
        direction = np.array([1.0, 0.0, 0.0]) - normal_axis * normal_axis[0]

    direction = direction - np.dot(direction, normal_axis) * normal_axis  # project into tangent plane
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        # direction happened to be ~parallel to normal_axis - pick any in-plane fallback
        fallback = np.array([1.0, 0.0, 0.0]) if abs(normal_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        direction = fallback - np.dot(fallback, normal_axis) * normal_axis
        norm = np.linalg.norm(direction)
    return direction / norm


def compute_arch_center(mesh, labels):
    """Centroid of all tooth-labeled (non-gum) face centroids - a stable whole-arch reference
    point for telling each tooth's facial (outward) direction from its lingual (inward) one."""
    tooth_idx = np.where(labels != 0)[0]
    return mesh.triangles_center[tooth_idx].mean(axis=0)


def tooth_facial_direction(origin, normal_axis, tangent_axis, arch_center):
    """
    Unit direction pointing facially (buccal/labial - outward, away from the arch's center) at
    this tooth's position, in its local tangent plane (perpendicular to both normal_axis and
    tangent_axis, i.e. the binormal_axis rectangular_footprint would derive internally). The
    lingual direction is simply the negation of this.
    """
    outward = origin - arch_center
    outward = outward - np.dot(outward, normal_axis) * normal_axis
    outward = outward - np.dot(outward, tangent_axis) * tangent_axis
    return outward / np.linalg.norm(outward)


def rectangular_footprint(mesh, origin, normal_axis, tangent_axis, width_mm, height_mm, slab_mm,
                           min_occlusality=0.6, min_component_faces=5):
    """
    Boolean mask of the single connected component of eligible faces closest to `origin`, where a
    face is eligible if its centroid lies within a width_mm x height_mm rectangle in the local
    tangent plane at `origin` (spanned by tangent_axis and binormal_axis = normal_axis x
    tangent_axis), within slab_mm of that plane along normal_axis, AND its own face normal is
    aligned with normal_axis (dot product >= min_occlusality) - simulating a scanner sensor held
    normal to the local surface, capturing the local surface facing that direction (the occlusal
    top when normal_axis is a tooth's own occlusal normal; a side wall when it's a facial/lingual
    direction instead) plus immediately adjacent gum, but not surfaces facing a meaningfully
    different way (a real capture wouldn't see those without reorienting to look elsewhere).

    This picks a single connected component rather than taking every eligible face directly: a
    tooth's cusps and ridges mean a face on the far/hidden side of a ridge can be geometrically
    close enough to land in the same linear (u, v, w) box as the visible surface, even though a
    real scanner looking straight down could never see it (occluded by the ridge itself) and it
    is only "connected" to the visible patch via a path that leaves the box first - eligible faces
    reliably fragment into many components (dozens, for a facial/lingual sweep across a whole
    tooth) separated by exactly these occluded-path gaps.

    It also does NOT simply flood-fill from the single closest-to-origin eligible face: that face
    can itself land in a tiny disconnected sliver (a stray face right at `origin` whose immediate
    neighbors fall just outside the box or alignment cutoff), stranding the search away from the
    real surface entirely. Components smaller than min_component_faces are ignored when picking
    the closest one, and ties/near-ties favor whichever component is actually nearest.
    """
    binormal_axis = np.cross(normal_axis, tangent_axis)
    binormal_axis = binormal_axis / np.linalg.norm(binormal_axis)
    tangent_axis = np.cross(binormal_axis, normal_axis)  # re-orthogonalize after the cross products

    rel = mesh.triangles_center - origin
    u = rel @ tangent_axis
    v = rel @ binormal_axis
    w = rel @ normal_axis
    occlusality = mesh.face_normals @ normal_axis
    eligible = ((np.abs(u) <= width_mm / 2) & (np.abs(v) <= height_mm / 2) & (np.abs(w) <= slab_mm / 2)
                & (occlusality >= min_occlusality))

    result = np.zeros(len(mesh.faces), dtype=bool)
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        return result

    a, b = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    keep = eligible[a] & eligible[b]
    a, b = a[keep], b[keep]
    n = len(mesh.faces)
    adj = csr_matrix((np.ones(len(a) * 2), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
    n_comp, comp_labels = connected_components(adj[eligible_idx][:, eligible_idx], directed=False)

    dist_to_origin = np.sum(rel[eligible_idx] ** 2, axis=1)
    sizes = np.bincount(comp_labels)
    valid_comps = np.where(sizes >= min_component_faces)[0]
    if len(valid_comps) == 0:
        valid_comps = np.arange(n_comp)  # nothing meets the size floor - fall back to whatever exists
    valid_mask = np.isin(comp_labels, valid_comps)
    best_local_idx = np.where(valid_mask)[0][np.argmin(dist_to_origin[valid_mask])]
    chosen_comp = comp_labels[best_local_idx]

    result[eligible_idx[comp_labels == chosen_comp]] = True
    return result


def sweep_footprints(mesh, labels, present, tooth_order, filter_axis_fn, width_mm, height_mm, slab_mm,
                       min_alignment, accumulated):
    """
    Shared sweep loop: for each tooth in tooth_order, capture a rectangular_footprint filtered by
    whatever local axis filter_axis_fn(origin, normal_axis, tangent_axis) returns (the tooth's own
    occlusal normal for an occlusal sweep, or a facial/lingual direction for a wall sweep), OR the
    tooth's own normal_axis (occlusal sweep). Returns the per-stage accumulated masks and the
    final accumulated mask (so a caller can chain further sweeps on top).
    """
    masks = []
    for tooth_label in tooth_order:
        origin, normal_axis = tooth_local_frame(mesh, labels, tooth_label)
        tangent_axis = tooth_tangent_direction(mesh, labels, tooth_label, present, normal_axis)
        filter_axis = filter_axis_fn(origin, normal_axis, tangent_axis)
        footprint = rectangular_footprint(mesh, origin, filter_axis, tangent_axis, width_mm, height_mm, slab_mm,
                                           min_alignment)
        accumulated = accumulated | footprint
        masks.append(accumulated.copy())
    return masks, accumulated


def grow_patch_scanner_sweep(mesh, labels, seed_label, rng, width_mm=14.0, height_mm=14.0, slab_mm=16.0,
                              min_occlusality=0.6):
    """
    Simulate a scanner sweeping tooth-by-tooth outward from seed_label (see
    order_teeth_from_seed), capturing one rectangular local-frame footprint per tooth position
    (see rectangular_footprint) and accumulating them - i.e. what the intraoral scanner actually
    sees as it moves along the arch, rather than a flood-fill blob grown from a single global axis.

    width_mm/height_mm default to ~14x14 (~196mm²), matching the live scanner's measured
    per-snapshot footprint of ~174mm² / one tooth (see MAA_ToDo.md). slab_mm bounds how far a
    face may sit from the local tangent plane; min_occlusality (see rectangular_footprint) keeps
    the capture occlusal-first - the tooth top and its immediately surrounding gum, not the
    buccal/lingual walls further down. This is phase 1 of grow_patch_full_sweep in isolation.

    Returns (masks, tooth_order): masks[i] is the accumulated footprint through tooth_order[i],
    each a superset of the last.
    """
    present = set(np.unique(labels).tolist()) & set(TOOTH_LABELS)
    order = order_teeth_from_seed(seed_label, present, rng)
    masks, _ = sweep_footprints(mesh, labels, present, order, lambda o, n, t: n, width_mm, height_mm, slab_mm,
                                  min_occlusality, np.zeros(len(labels), dtype=bool))
    return masks, order


def grow_patch_full_sweep(mesh, labels, seed_label, rng, width_mm=14.0, height_mm=14.0, slab_mm=16.0,
                           min_alignment=0.2):
    """
    The full three-phase scan simulation:
      1. Occlusal coverage tooth-by-tooth from the seed incisor, one side of the arch swept to
         completion before the other (grow_patch_scanner_sweep).
      2. Facial (buccal/labial) wall added in ONE continuous sweep across the whole arch, from
         one back tooth through to the other (order_arch_from_one_end) - not restarting from the
         front seed, since this phase only begins once occlusal has covered every tooth.
      3. Lingual wall added the same way, over the same tooth order as the facial sweep.

    Each phase's sweep starts fresh at that phase's own first tooth but accumulates on top of
    everything already captured - phases never remove or replace earlier coverage, only add to it.

    Returns (masks, stages): masks[i] is the accumulated patch after stage i; stages[i] is
    (phase, tooth_label) identifying what stage i added, phase in {"occlusal", "facial", "lingual"}.
    """
    present = set(np.unique(labels).tolist()) & set(TOOTH_LABELS)
    accumulated = np.zeros(len(labels), dtype=bool)

    occlusal_order = order_teeth_from_seed(seed_label, present, rng)
    occlusal_masks, accumulated = sweep_footprints(mesh, labels, present, occlusal_order, lambda o, n, t: n,
                                                      width_mm, height_mm, slab_mm, min_alignment, accumulated)

    wall_order = order_arch_from_one_end(present, rng)
    arch_center = compute_arch_center(mesh, labels)
    facial_masks, accumulated = sweep_footprints(
        mesh, labels, present, wall_order, lambda o, n, t: tooth_facial_direction(o, n, t, arch_center),
        width_mm, height_mm, slab_mm, min_alignment, accumulated)
    lingual_masks, accumulated = sweep_footprints(
        mesh, labels, present, wall_order, lambda o, n, t: -tooth_facial_direction(o, n, t, arch_center),
        width_mm, height_mm, slab_mm, min_alignment, accumulated)

    masks = occlusal_masks + facial_masks + lingual_masks
    stages = ([("occlusal", t) for t in occlusal_order]
              + [("facial", t) for t in wall_order]
              + [("lingual", t) for t in wall_order])
    return masks, stages


def compute_occlusality(mesh, occlusal_axis):
    """
    Per-face occlusality = dot(face_normal, occlusal_axis), in [-1, 1]. High = face points
    straight at the scanner (an occlusal cap); low/negative = face points sideways (a
    buccal/lingual wall) or away (the underside).
    """
    return mesh.face_normals @ occlusal_axis


def grow_patch_coverage(mesh, extent_mask, occlusality, occlusal_threshold, labels):
    """
    Best-first flood fill restricted to extent_mask, seeded at the extent's highest-occlusality
    TOOTH face (gum, label 0, is excluded from seed selection - see below) and expanding to
    extent-neighbors in descending-occlusality order. Growth stops once no remaining frontier
    candidate meets occlusal_threshold.

    This intentionally does NOT threshold-everywhere-then-take-the-largest-connected-component:
    real front teeth (thin incisal edges, not flat occlusal tables like molars) fragment into
    many tiny disconnected high-occlusality islands under a naive threshold, separated by
    low-occlusality fissures/valleys - the "largest" surviving island at a restrictive threshold
    is often an incidental patch of gum rather than any tooth's occlusal surface. Growing from a
    single best seed guarantees a single connected region by construction, and for a fixed seed
    a lower threshold's result is always a superset of a higher threshold's (same search
    trajectory, cut off later), so a coverage sweep is a real growth sequence.

    The seed is restricted to tooth faces because, especially near incisors (thin labial face +
    narrow incisal edge, not a flat table), the single globally-highest-occlusality face in an
    extent can genuinely be a flat patch of gum rather than anything on the seed's own tooth -
    gum can still be admitted once growth reaches it, it just shouldn't be where growth starts.

    A HIGH occlusal_threshold (near the top of occlusality's range) is restrictive: occlusal
    caps only. A LOW/negative occlusal_threshold is permissive: admits buccal/lingual walls too.
    """
    extent_idx = np.where(extent_mask)[0]
    result = np.zeros(len(extent_mask), dtype=bool)
    if len(extent_idx) == 0:
        return result
    tooth_idx = extent_idx[labels[extent_idx] != 0]
    seed_candidates = tooth_idx if len(tooth_idx) > 0 else extent_idx
    seed_face_idx = seed_candidates[np.argmax(occlusality[seed_candidates])]
    if occlusality[seed_face_idx] < occlusal_threshold:
        return result  # nothing in the extent clears the threshold at all

    adjacency = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        if extent_mask[a] and extent_mask[b]:
            adjacency[a].append(b)
            adjacency[b].append(a)

    visited = np.zeros(len(mesh.faces), dtype=bool)
    heap = [(-occlusality[seed_face_idx], int(seed_face_idx))]
    visited[seed_face_idx] = True

    while heap:
        neg_occ, face_idx = heapq.heappop(heap)
        if -neg_occ < occlusal_threshold:
            break
        result[face_idx] = True
        for nb in adjacency[face_idx]:
            if not visited[nb]:
                visited[nb] = True
                heapq.heappush(heap, (-occlusality[nb], nb))

    return result
