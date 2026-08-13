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
    def __init__(self, num_classes=17, feature_dim=24, k=32, dilation_ks=(200, 600, 1800),
                 area_thresholds=(40.0, 180.0, 360.0), dilation_gating=True):
        """
        :param num_classes: Number of classes to predict.
        :param feature_dim: Input per-face feature width (24 for the PatchPreTransform layout).
        :param k: Neighbor count used by every edge-conv block (local and dilated alike) - also
            the single source of truth models/patch_collate.py reads to precompute matching
            neighbor indices.
        :param dilation_ks: Candidate-neighborhood sizes for the 3 dilated blocks, before FPS
            samples k of them - same role as the paper's 200/600/1800.
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

    def forward(self, x, pos, area_mm2=None, local_idx=None, dilated_idx=(None, None, None)):
        x1, _ = self.edge_graph_conv_block1(x, pos, idx=local_idx)
        x2, _ = self.edge_graph_conv_block2(x1)
        x3, _ = self.edge_graph_conv_block3(x2)

        x = torch.cat([x1, x2, x3], dim=2)
        x = self.local_hidden_layer(x)

        gate1, gate2, gate3 = self._resolve_gates(area_mm2)
        idx_fps1, idx_fps2, idx_fps3 = dilated_idx

        d1 = self.dilated_edge_graph_conv_block1(x, pos, idx_fps=idx_fps1)[0] if gate1 else x
        d2 = self.dilated_edge_graph_conv_block2(d1, pos, idx_fps=idx_fps2)[0] if gate2 else d1
        d3 = self.dilated_edge_graph_conv_block3(d2, pos, idx_fps=idx_fps3)[0] if gate3 else d2

        x = torch.cat([x, d1, d2, d3], dim=2)
        x = self.global_hidden_layer(x)

        x = self.feature_importance(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        return x
