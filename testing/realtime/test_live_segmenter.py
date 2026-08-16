import argparse
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork  # noqa: E402
from testing.realtime.mesh_viewer_segmented import LiveSegmenter, colors_for_predictions, flat_shaded_mesh  # noqa: E402

# Headless test of the actual live-inference pipeline - no Open3D window, no network, no real
# scanner - so this can run in this environment despite having neither the real hardware nor a
# display. Uses a raw (dense, un-downsampled) Teeth3DS mesh, split into chunks the way the REAL
# scanner was confirmed to behave (see TRAINING_CONCERNS.md / the live debug log this was built
# from): every mesh_id is NEW and permanent - never grown or replaced in place. An earlier version
# of this test simulated one mesh_id growing across several calls (matching what the code
# ASSUMED before ever touching real hardware); that pattern turned out not to exist in practice,
# so it's been replaced here with what's actually real: many small, distinct, non-overlapping
# chunks for the main growth check, plus a dedicated test for genuinely OVERLAPPING chunks (the
# thing that actually broke against the real scanner - see live_preprocessing.py's
# claim_new_voxels).


def _spatial_chunks(mesh, n_chunks):
    """Non-overlapping face-index groups ordered by spatial position (PCA axis), matching
    fake_mesh_server.py's _build_regions - real scanner chunks are spatially coherent, not random
    face subsets (verified separately: a random subset creates pathologically fragmented geometry
    pyfqmr's decimator handles badly)."""
    centroids = mesh.triangles_center
    centered = centroids - centroids.mean(axis=0)
    axis_pos = centered @ np.linalg.svd(centered, full_matrices=False)[2][0]
    order = np.argsort(axis_pos)
    return np.array_split(order, n_chunks)


def _chunk_from_faces(mesh, face_idx):
    faces = mesh.faces[face_idx]
    used_v = np.unique(faces)
    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[used_v] = np.arange(len(used_v))
    return mesh.vertices[used_v], remap[faces]


def main(obj_path, ckpt_path, num_classes, device):
    print(f"=== loading real mesh: {obj_path} ===")
    mesh = trimesh.load(obj_path, process=False)
    print(f"  {len(mesh.faces)} faces, area={mesh.area:.1f}mm2, "
          f"native density={len(mesh.faces) / mesh.area:.1f} faces/mm2")

    print(f"\n=== loading checkpoint {ckpt_path} on {device} ===")
    model = PatchLitDilatedToothSegmentationNetwork.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    segmenter = LiveSegmenter(model, device, num_classes=num_classes, target_density=5.0, arch="lower")

    print("\n=== process_once() with no chunks yet: should return None, not crash ===")
    assert segmenter.process_once() is None
    print("  ok")

    print("\n=== growth: distinct, non-overlapping, NEVER-updated mesh_ids (the real pattern) ===")
    chunk_splits = _spatial_chunks(mesh, 6)
    total_raw_faces = 0
    for i, face_idx in enumerate(chunk_splits):
        vertices, triangles = _chunk_from_faces(mesh, face_idx)
        segmenter.update_chunk(mesh_id=i, vertices=vertices, triangles=triangles)
        total_raw_faces += len(face_idx)

        result = segmenter.process_once()
        assert result is not None
        down, colors, label_positions, stats = result
        density = stats["down_faces"] / stats["raw_area_mm2"]
        print(f"  step {i}: raw={stats['raw_faces']:6d} decimated={stats['decimated_faces']:5d} "
              f"down={stats['down_faces']:5d} (density={density:.2f}/mm2)  "
              f"downsample={stats['t_downsample']:.2f}s collate={stats['t_collate']:.2f}s "
              f"infer={stats['t_infer']:.2f}s  labels={[t for _, t in label_positions]}")
        assert colors.shape == (len(down.faces), 3)
        assert colors.dtype == np.uint8
        assert np.all((colors >= 0) & (colors <= 255))
        assert stats["raw_faces"] == total_raw_faces
        # non-overlapping chunks: dedup should cost only a small fraction (grid-aliasing between
        # genuinely-adjacent faces, not real redundancy) - not the ~30-50% seen when the voxel
        # size was first (wrongly) derived directly from target_density
        loss_frac = 1 - stats["down_faces"] / stats["decimated_faces"]
        assert loss_frac < 0.15, f"non-overlapping chunks lost {loss_frac:.1%} to dedup - voxel grid too coarse"
        assert 3.5 < density < 6.0, f"downsampled density {density:.2f} should land near target 5.0"
        for centroid, text in label_positions:
            assert centroid.shape == (3,)
            assert text.startswith("#")
            universal = int(text[1:])
            assert 17 <= universal <= 32, f"a lower-arch tooth number should be 17-32, got {universal}"

    print("\n=== flat_shaded_mesh: per-face coloring produces 3x vertices, no sharing ===")
    vertices, triangles, vcolors = flat_shaded_mesh(down, colors)
    assert vertices.shape[0] == len(down.faces) * 3
    assert triangles.shape == (len(down.faces), 3)
    assert vcolors.shape[0] == len(down.faces) * 3
    assert np.array_equal(triangles, np.arange(len(vertices)).reshape(-1, 3)), "each face must own unique vertices"
    print(f"  ok - {len(down.faces)} faces -> {vertices.shape[0]} unique (non-shared) vertices")

    print("\n=== dedicated overlap test: a NEW mesh_id re-observing already-covered surface ===")
    # this is exactly what broke against the real scanner: normal scanning motion re-visits
    # surface already captured under an earlier (permanent, never-updated) mesh_id
    total_faces_before = stats["down_faces"]
    overlap_face_idx = chunk_splits[0]  # the exact same real-world surface as mesh_id=0, already sent
    vertices, triangles = _chunk_from_faces(mesh, overlap_face_idx)
    segmenter.update_chunk(mesh_id=999, vertices=vertices, triangles=triangles)
    result = segmenter.process_once()
    down3, colors3, label_positions3, stats3 = result
    new_faces = stats3["down_faces"] - total_faces_before
    print(f"  before: {total_faces_before} faces. After re-sending mesh_id=0's exact surface as a "
          f"NEW mesh_id=999: {stats3['down_faces']} faces ({new_faces:+d})")
    # a full re-observation of already-covered surface should add close to nothing - some small
    # positive drift is fine (voxel-grid boundary effects), but nowhere near the ~1/6 of the total
    # that chunk's own raw size would suggest if dedup weren't working at all
    assert new_faces < 0.05 * total_faces_before, \
        f"re-observing already-covered surface added {new_faces} faces - dedup isn't catching real overlap"
    print("  ok - dedup correctly suppressed a redundant re-observation")

    print("\n=== a genuinely NEW, disjoint mesh_id still contributes real new faces ===")
    # take faces far from anything sent so far, offset in space so it's unambiguously disjoint
    fresh_idx = chunk_splits[-1][:300]
    vertices2, triangles2 = _chunk_from_faces(mesh, fresh_idx)
    segmenter.update_chunk(mesh_id=1000, vertices=vertices2 + np.array([200.0, 0, 0]), triangles=triangles2)
    result2 = segmenter.process_once()
    down4, colors4, label_positions4, stats4 = result2
    print(f"  faces grew from {stats3['down_faces']} to {stats4['down_faces']} after a disjoint new chunk")
    assert stats4["down_faces"] > stats3["down_faces"], "a genuinely new, disjoint chunk should add real faces"

    print("\n=== max_model_faces cap: merged mesh must never exceed it, even from one huge chunk ===")
    # Real-hardware testing found voxel dedup alone doesn't bound total merged size over a long
    # scan (periodic large re-batched chunks + continuous small-chunk overlap both leak some "new"
    # faces past dedup) - the merged mesh grew past 160k faces and the model's on-the-fly O(N^2)
    # cdist (models/patch_layer.py's EdgeGraphConvBlock 2/3) tried to allocate 100+ GiB. This
    # checks the cap added in process_once() actually holds regardless of how much raw geometry
    # comes in at once.
    capped_segmenter = LiveSegmenter(model, device, num_classes=num_classes, target_density=5.0,
                                      arch="lower", max_model_faces=500)
    capped_segmenter.update_chunk(mesh_id=0, vertices=mesh.vertices, triangles=mesh.faces)
    result_capped = capped_segmenter.process_once()
    down_capped, _, _, stats_capped = result_capped
    print(f"  fed the entire {len(mesh.faces)}-face mesh as one chunk with max_model_faces=500: "
          f"dedup_faces={stats_capped['dedup_faces']} down_faces={stats_capped['down_faces']}")
    assert stats_capped["down_faces"] <= 500, \
        f"cap should hold down_faces <= max_model_faces=500, got {stats_capped['down_faces']}"
    assert stats_capped["dedup_faces"] > stats_capped["down_faces"], \
        "dedup_faces should exceed down_faces when the cap actually triggered"
    print("  ok - cap held regardless of how much geometry arrived at once")

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless test of the live-inference processing core")
    parser.add_argument("--obj_path",
                         default="/home/hesy-tech/MAA/IO_Scanner-Dataset_Models/data/Teeth3DS/"
                                 "lower/JJ19KE8W/JJ19KE8W_lower.obj")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.obj_path, args.ckpt, args.num_classes, args.device)
