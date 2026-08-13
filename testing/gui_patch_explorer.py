import argparse
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import (TOOTH_LABELS, sweep_footprints, compute_arch_center, load_cached_arch,
                                      order_arch_from_one_end, order_teeth_from_seed, pick_incisor_seed_label,
                                      tooth_facial_direction)
from testing.visualize_patch_growth import GUM_COLOR, local_view_axes
from utils.teeth_numbering import coarse_label_to_colors, label_to_colors, label_to_coarse_label

BACKGROUND_COLOR = np.array([205, 205, 208], dtype=np.uint8)
GUM_COLOR_255 = (GUM_COLOR * 255).astype(np.uint8)


def build_phase_sequences(mesh, labels, seed_label, rng, width_mm, height_mm, slab_mm, min_occlusality):
    """
    Three INDEPENDENT stage sequences (occlusal, facial, lingual), each its own sweep starting
    from an empty accumulator - unlike grow_patch_full_sweep, these are NOT gated on each other
    (facial doesn't wait for occlusal to finish), since a live explorer should let each trackpad
    be moved freely on its own to inspect that phase in isolation.
    """
    present = set(np.unique(labels).tolist()) & set(TOOTH_LABELS)
    empty = np.zeros(len(labels), dtype=bool)

    occlusal_order = order_teeth_from_seed(seed_label, present, rng)
    occlusal_masks, _ = sweep_footprints(mesh, labels, present, occlusal_order, lambda o, n, t: n,
                                           width_mm, height_mm, slab_mm, min_occlusality, empty)

    wall_order = order_arch_from_one_end(present, rng)
    arch_center = compute_arch_center(mesh, labels)
    facial_masks, _ = sweep_footprints(
        mesh, labels, present, wall_order, lambda o, n, t: tooth_facial_direction(o, n, t, arch_center),
        width_mm, height_mm, slab_mm, min_occlusality, empty)
    lingual_masks, _ = sweep_footprints(
        mesh, labels, present, wall_order, lambda o, n, t: -tooth_facial_direction(o, n, t, arch_center),
        width_mm, height_mm, slab_mm, min_occlusality, empty)

    return occlusal_masks, facial_masks, lingual_masks


def build_pyvista_mesh(mesh):
    faces = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]).astype(np.int64)
    return pv.PolyData(mesh.vertices, faces.flatten())


def colors_for_mask(mask, labels, num_classes):
    colors = np.tile(BACKGROUND_COLOR, (len(labels), 1))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return colors
    if num_classes == 5:
        colors[idx] = coarse_label_to_colors(label_to_coarse_label(labels[idx])).astype(np.uint8)
    else:
        colors[idx] = label_to_colors(labels[idx]).astype(np.uint8)
        colors[idx[labels[idx] == 0]] = GUM_COLOR_255
    return colors


def main(pt_path, seed, num_classes, width_mm, height_mm, slab_mm, min_occlusality):
    mesh, labels = load_cached_arch(pt_path)
    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)

    occlusal_masks, facial_masks, lingual_masks = build_phase_sequences(
        mesh, labels, seed_label, rng, width_mm, height_mm, slab_mm, min_occlusality)
    n_occ, n_fac, n_lin = len(occlusal_masks), len(facial_masks), len(lingual_masks)

    pv_mesh = build_pyvista_mesh(mesh)
    pv_mesh.cell_data["colors"] = colors_for_mask(np.zeros(len(labels), dtype=bool), labels, num_classes)

    state = {"occ": 0, "fac": 0, "lin": 0}

    plotter = pv.Plotter(title=f"Patch explorer - {os.path.basename(pt_path)} - seed tooth {seed_label}")
    plotter.set_background("white")
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, show_edges=False, smooth_shading=False)
    info_actor = plotter.add_text("", position="lower_left", font_size=10, color="black")

    def refresh():
        nonlocal info_actor
        mask = np.zeros(len(labels), dtype=bool)
        if state["occ"] > 0:
            mask |= occlusal_masks[state["occ"] - 1]
        if state["fac"] > 0:
            mask |= facial_masks[state["fac"] - 1]
        if state["lin"] > 0:
            mask |= lingual_masks[state["lin"] - 1]
        pv_mesh.cell_data["colors"] = colors_for_mask(mask, labels, num_classes)
        n_faces = int(mask.sum())
        area = mesh.area_faces[mask].sum()
        plotter.remove_actor(info_actor)
        info_actor = plotter.add_text(
            f"occlusal {state['occ']}/{n_occ}   facial {state['fac']}/{n_fac}   lingual {state['lin']}/{n_lin}\n"
            f"{n_faces} faces, {area:.0f} mm2",
            position="lower_left", font_size=10, color="black")
        plotter.render()

    def make_callback(key):
        def callback(value):
            state[key] = int(round(value))
            refresh()
        return callback

    plotter.add_slider_widget(make_callback("occ"), [0, n_occ], value=0, title="Occlusal",
                               pointa=(0.05, 0.94), pointb=(0.31, 0.94), style="modern", color="#3f7fbf")
    plotter.add_slider_widget(make_callback("fac"), [0, n_fac], value=0, title="Facial",
                               pointa=(0.38, 0.94), pointb=(0.64, 0.94), style="modern", color="#bf7f3f")
    plotter.add_slider_widget(make_callback("lin"), [0, n_lin], value=0, title="Lingual",
                               pointa=(0.71, 0.94), pointb=(0.97, 0.94), style="modern", color="#7fbf3f")

    def reset_all():
        state["occ"] = state["fac"] = state["lin"] = 0
        refresh()

    plotter.add_key_event("r", reset_all)

    # start with a top-down view fit to the whole arch, matching the static visualizations
    tangent_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, np.ones(len(labels), dtype=bool))
    center = mesh.vertices.mean(axis=0)
    radius = float(np.linalg.norm(mesh.extents))
    plotter.camera_position = [
        (center + occlusal_axis * radius * 1.5).tolist(),
        center.tolist(),
        secondary_axis.tolist(),
    ]

    refresh()
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 3D explorer: three sliders (occlusal/facial/"
                                                   "lingual) live-color the mesh as each phase's sweep is revealed")
    parser.add_argument("--pt_path", required=True, help="Path to a cached data_*.pt file")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the incisor seed tooth and sweep sides")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    parser.add_argument("--width_mm", type=float, default=14.0)
    parser.add_argument("--height_mm", type=float, default=14.0)
    parser.add_argument("--slab_mm", type=float, default=16.0)
    parser.add_argument("--min_occlusality", type=float, default=0.6)
    args = parser.parse_args()
    main(args.pt_path, args.seed, args.num_classes, args.width_mm, args.height_mm, args.slab_mm,
         args.min_occlusality)
