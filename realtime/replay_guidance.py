import argparse
import glob
import os

import numpy as np

# mesh_viewer_segmented.py is a sibling in this same realtime/ directory - bare import (this file
# moved from repo root into realtime/ alongside it, so no sys.path hack is needed anymore: running
# this directly auto-adds realtime/ to sys.path, same as mesh_viewer_segmented.py's own bare
# `import mesh_wire`/`from debug_log import logger`). Imported (not reimplemented) so this actually
# replays the real code path LiveSegmenter runs, not a maybe-drifted copy of it.
from mesh_viewer_segmented import _GUIDANCE_LEFT_SEQUENCE, _GUIDANCE_RIGHT_SEQUENCE, update_guidance_confirmation

# New, additive-only script - replays a SAVED live-hardware run's real per-cycle predictions
# (realtime/mesh_viewer_segmented.py's snapshots) through the OLD (stateless, single-cycle
# threshold crossing) vs NEW (Finding #13, REALTIME.md - sustained multi-cycle confirmation)
# guidance-confirmation logic side by side, to validate the fix against real data BEFORE trusting
# it on live hardware. Reuses the actual update_guidance_confirmation function (imported, not
# reimplemented) for the NEW logic, so this is testing the real code path, not a reimplementation
# of it that could silently drift from what LiveSegmenter actually runs.
#
# Doesn't replay full _compute_guidance (target position/distance/compass) - that needs
# self._last_chunk_centroid, which isn't saved in a snapshot (it's derived from the newest raw
# chunk's own centroid, not the accumulated mesh). What IS replayable and is the actual point of
# this validation: WHEN each label gets confirmed under each scheme, and WHEN the sweep
# transitions/regimen-complete would fire as a result - since that's driven purely by which labels
# are in the confirmed set, independent of position.


def old_logic_confirmed(counts, min_faces):
    """The ORIGINAL behavior being replaced: whatever crosses the threshold THIS cycle, with zero
    memory of any other cycle - exactly what _compute_guidance did before Finding #13."""
    return {lbl for lbl in range(1, 17) if lbl < len(counts) and counts[lbl] >= min_faces}


def first_missing(seq, confirmed):
    for lbl in seq:
        if lbl not in confirmed:
            return lbl
    return None


def regimen_target(confirmed):
    """Mirrors _compute_guidance's own sequencing (minus the Finding #11 center-return gate, which
    depends on scanner position data not available here) - which sweep/target the regimen would be
    on given a confirmed-label set alone."""
    target = first_missing(_GUIDANCE_LEFT_SEQUENCE, confirmed)
    if target is not None:
        return 'LEFT', target
    target = first_missing(_GUIDANCE_RIGHT_SEQUENCE, confirmed)
    if target is not None:
        return 'RIGHT', target
    return 'DONE', None


def main():
    parser = argparse.ArgumentParser(description="Replay OLD vs NEW (Finding #13) guidance confirmation "
                                                   "logic against a saved live-hardware run")
    parser.add_argument("run_dir", help="realtime/logs/snapshots/<run_id>/ directory")
    parser.add_argument("--min_faces", type=int, default=30)
    parser.add_argument("--confirm_cycles", type=int, default=8,
                         help="must match realtime/mesh_viewer_segmented.py's "
                              "LiveSegmenter guidance_confirm_cycles default (8) to replay what a "
                              "live run would actually do - override to test a candidate value "
                              "before changing that default")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "*.npz")),
                    key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    print(f"{len(files)} cycles in {args.run_dir}\n")

    streaks, new_confirmed = {}, set()
    old_first_seen, new_first_seen = {}, {}
    old_target_log, new_target_log = [], []
    old_prev, new_prev = None, None

    for f in files:
        d = np.load(f)
        cycle = int(d["cycle"])
        counts = np.bincount(d["pred_labels"], minlength=17)

        old_conf = old_logic_confirmed(counts, args.min_faces)
        for lbl in old_conf:
            old_first_seen.setdefault(lbl, cycle)

        update_guidance_confirmation(counts, streaks, new_confirmed, args.min_faces, args.confirm_cycles)
        for lbl in new_confirmed:
            new_first_seen.setdefault(lbl, cycle)

        old_stage, old_tgt = regimen_target(old_conf)
        new_stage, new_tgt = regimen_target(new_confirmed)
        if (old_stage, old_tgt) != old_prev:
            old_target_log.append((cycle, old_stage, old_tgt))
            old_prev = (old_stage, old_tgt)
        if (new_stage, new_tgt) != new_prev:
            new_target_log.append((cycle, new_stage, new_tgt))
            new_prev = (new_stage, new_tgt)

    print("=== first cycle each label is CONFIRMED: OLD (single-cycle) vs NEW (sustained) ===")
    for lbl in range(1, 17):
        o = old_first_seen.get(lbl, "never")
        n = new_first_seen.get(lbl, "never")
        flag = "  <-- DELAYED (noise filtered)" if isinstance(o, int) and isinstance(n, int) and n > o else ""
        print(f"  label {lbl:2d}: old={o!s:>6}  new={n!s:>6}{flag}")

    print("\n=== regimen stage/target transitions: OLD logic (stateless, prone to flicker) ===")
    for cycle, stage, tgt in old_target_log:
        print(f"  cycle {cycle:3d}: {stage:5s} target={tgt}")

    print("\n=== regimen stage/target transitions: NEW logic (Finding #13, sustained confirmation) ===")
    for cycle, stage, tgt in new_target_log:
        print(f"  cycle {cycle:3d}: {stage:5s} target={tgt}")

    old_flickers = len(old_target_log) - 1
    new_flickers = len(new_target_log) - 1
    print(f"\nstage/target transitions: old={old_flickers}  new={new_flickers}")


if __name__ == "__main__":
    main()
