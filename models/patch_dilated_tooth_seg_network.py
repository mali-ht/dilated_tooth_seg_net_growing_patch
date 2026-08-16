import time

import torch
from torch import nn

from models.patch_layer import (BasicPointLayer, DilatedEdgeGraphConvBlock, EdgeGraphConvBlock,
                                 PointFeatureImportance, ResidualBasicPointLayer)

# Component 4 (claude_code_prompt.md) model surgery, adapted from
# models/dilated_tooth_seg_network.py (untouched). Same channel architecture as the original
# (24-dim local blocks, 60-dim dilated blocks, 1024/512/256 head) - this is a surgery, not a
# redesign, so the paper's validated backbone shape is preserved. Four changes:
#
# 1. STNkd removed from the forward path entirely (not even constructed) - claude_code_prompt.md
#    Component 4.
# 2. All blocks come from models/patch_layer.py (GroupNorm instead of BatchNorm, small-N safe).
# 3. Variable N: no pad/resample-to-16,000 assumption anywhere - N is just whatever the patch is.
# 4. Area-gated dilation: each of the 3 dilated blocks only runs if the patch's real mm^2 area
#    clears that block's threshold (area_thresholds, default ~40/180/360mm^2 per
#    claude_code_prompt.md); a gated-off block contributes its OWN INPUT unchanged (identity) to
#    both the next block in the chain and the final concat, so global_hidden_layer's expected
#    240-channel input (60 base + 3x60 dilated) stays valid regardless of how many blocks ran.
#    dilation_gating=False disables gating entirely (all blocks always run) - the ungated
#    baseline from claude_code_prompt.md's Component 5 experiment matrix.
#
# forward() accepts optional precomputed neighbor indices (local_idx for block 1, dilated_idx for
# the 3 dilated blocks) - see models/patch_collate.py, which computes these on CPU ahead of time
# for the metric-space-only graphs. Omitting them falls back to on-the-fly computation (slower,
# but makes this network usable/testable standalone without the collate step).


class PatchDilatedToothSegmentationNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, k=32, dilation_ks=(200, 900, 1800),
                 area_thresholds=(40.0, 180.0, 360.0), dilation_gating=True):
        """
        :param num_classes: Number of classes to predict.
        :param feature_dim: Input per-face feature width (24 for the PatchPreTransform layout).
        :param k: Neighbor count used by every edge-conv block (local and dilated alike) - also
            the single source of truth models/patch_collate.py reads to precompute matching
            neighbor indices.
        :param dilation_ks: Candidate-neighborhood sizes for the 3 dilated blocks, before FPS
            samples k of them - matches the original paper/reference implementation's own
            200/900/1800 (models/dilated_tooth_seg_network.py:31-39, untouched) exactly. This
            default was previously (200, 600, 1800) - an undocumented, unintentional drift from
            the reference values with no design rationale behind it anywhere in this project's
            docs (confirmed: claude_code_prompt.md's Component 4 spec never asked for a
            dilation_k change, only norm/STN/variable-N/area-gating; TRAINING_CONCERNS.md had
            claimed 600 WAS the paper's own value, which is simply false against the actual
            reference file - see REALTIME.md's session log for the full trace). Restored to 900
            here; see that same log if a deliberate departure from the paper's value is wanted
            again later - it should be a documented decision, not a silent default.
        :param area_thresholds: mm^2 area a patch must reach for dilated blocks 1/2/3 to engage.
        :param dilation_gating: If False, all dilated blocks always run regardless of area.
        """
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.dilation_ks = tuple(dilation_ks)
        self.area_thresholds = tuple(area_thresholds)
        self.dilation_gating = dilation_gating

        self.edge_graph_conv_block1 = EdgeGraphConvBlock(in_channels=feature_dim, out_channels=24, k=k,
                                                           hidden_channels=24, edge_function="local_global")
        self.edge_graph_conv_block2 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=k,
                                                           hidden_channels=24, edge_function="local_global")
        self.edge_graph_conv_block3 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=k,
                                                           hidden_channels=24, edge_function="local_global")

        self.local_hidden_layer = BasicPointLayer(in_channels=24 * 3, out_channels=60)

        self.dilated_edge_graph_conv_block1 = DilatedEdgeGraphConvBlock(
            in_channels=60, hidden_channels=60, out_channels=60, k=k, dilation_k=self.dilation_ks[0],
            edge_function="local_global")
        self.dilated_edge_graph_conv_block2 = DilatedEdgeGraphConvBlock(
            in_channels=60, hidden_channels=60, out_channels=60, k=k, dilation_k=self.dilation_ks[1],
            edge_function="local_global")
        self.dilated_edge_graph_conv_block3 = DilatedEdgeGraphConvBlock(
            in_channels=60, hidden_channels=60, out_channels=60, k=k, dilation_k=self.dilation_ks[2],
            edge_function="local_global")

        self.global_hidden_layer = BasicPointLayer(in_channels=60 * 4, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

    def _resolve_gates(self, area_mm2):
        if not self.dilation_gating or area_mm2 is None:
            return True, True, True
        return tuple(area_mm2 >= t for t in self.area_thresholds)

    def forward(self, x, pos, area_mm2=None, local_idx=None, dilated_idx=(None, None, None), profile=False):
        """profile=False (the default, used everywhere in training) costs nothing extra - the
        live-inference debug investigation (REALTIME.md) is the only caller that ever passes
        profile=True. When it does, every stage is bracketed by torch.cuda.synchronize() (CUDA
        calls are queued asynchronously - without synchronizing, a naive time.time() around a GPU
        op measures how long it took to ENQUEUE the op, not how long it took to RUN, and nearly
        all the real time would falsely show up on whichever stage happens to be the first one
        that actually blocks on the result) and the per-stage wall times are stashed on
        self.last_timings (an instance attribute, not a second return value, so callers that don't
        care about profiling - i.e. every training call - are completely unaffected). The specific
        thing this exists to answer: block1 gets a precomputed idx (cheap), but block2/block3 do
        not (see this file's own module docstring + models/patch_layer.py's) - separating their
        timings from everything else confirms or denies how much of a live cycle's GPU time is
        that on-the-fly feature-space cdist specifically."""
        def _sync():
            if profile and x.is_cuda:
                torch.cuda.synchronize()

        timings = {} if profile else None
        _sync()

        t0 = time.time()
        x1, _ = self.edge_graph_conv_block1(x, pos, idx=local_idx)
        _sync()
        if profile:
            timings['block1_precomputed_idx'] = time.time() - t0

        t0 = time.time()
        x2, _ = self.edge_graph_conv_block2(x1)
        _sync()
        if profile:
            timings['block2_onthefly_cdist'] = time.time() - t0

        t0 = time.time()
        x3, _ = self.edge_graph_conv_block3(x2)
        _sync()
        if profile:
            timings['block3_onthefly_cdist'] = time.time() - t0

        t0 = time.time()
        x = torch.cat([x1, x2, x3], dim=2)
        x = self.local_hidden_layer(x)
        _sync()
        if profile:
            timings['local_concat_and_hidden'] = time.time() - t0

        gate1, gate2, gate3 = self._resolve_gates(area_mm2)
        self.last_gates = (gate1, gate2, gate3)  # instance attribute, same pattern as last_timings -
        # cheap (3 bools), always set regardless of profile= - lets a caller correlate "how much
        # dilated/global context did this prediction have" with area_mm2 (REALTIME.md's live
        # prediction-debugging methodology: a small patch with all gates off runs on local
        # features only, which plausibly explains locally-similar-looking neighbor teeth being
        # confused until enough area accumulates to engage wider context)
        idx_fps1, idx_fps2, idx_fps3 = dilated_idx

        t0 = time.time()
        d1 = self.dilated_edge_graph_conv_block1(x, pos, idx_fps=idx_fps1)[0] if gate1 else x
        _sync()
        if profile:
            timings['dilated_block1'] = time.time() - t0

        t0 = time.time()
        d2 = self.dilated_edge_graph_conv_block2(d1, pos, idx_fps=idx_fps2)[0] if gate2 else d1
        _sync()
        if profile:
            timings['dilated_block2'] = time.time() - t0

        t0 = time.time()
        d3 = self.dilated_edge_graph_conv_block3(d2, pos, idx_fps=idx_fps3)[0] if gate3 else d2
        _sync()
        if profile:
            timings['dilated_block3'] = time.time() - t0

        t0 = time.time()
        x = torch.cat([x, d1, d2, d3], dim=2)
        x = self.global_hidden_layer(x)
        x = self.feature_importance(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        _sync()
        if profile:
            timings['head'] = time.time() - t0
            self.last_timings = timings
        return x
