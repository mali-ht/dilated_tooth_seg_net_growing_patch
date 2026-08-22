import time

import torch
from torch import nn

from models.patch_layer import (BasicPointLayer, DilatedEdgeGraphConvBlock, EdgeGraphConvBlock,
                                 PointFeatureImportance, ResidualBasicPointLayer)
from models.patch_transformer_layer import GlobalTokenTransformerBlock

# New, additive-only file - models/patch_dilated_tooth_seg_network.py and
# models/patch_dilated_tooth_seg_transformer_network.py are both untouched. Adds a SECOND
# GlobalTokenTransformerBlock (models/patch_transformer_layer.py, reused unchanged - it's already
# generic over in_channels/embed_dim, so no new layer class was needed) operating on
# global_hidden_layer's fused 1024-dim local+dilated+early-global output, instead of the first
# block's raw 60-dim local-only input. User-directed 2026-08-22: keep the EARLY branch (on raw
# local features) AND add a LATE one (on the richer fused representation) rather than choosing
# between them - each token going into the late pass already carries fused local+dilated+
# early-global context before being globally cross-referenced a second time.
#
# The late block's output joins AFTER PointFeatureImportance, not before - so that gate keeps
# doing exactly what it always did (reweighting the local+dilated+early-global fusion) without
# changing its own width, and the late-global context is injected right where the residual
# refinement begins instead (res_block1's input channel count widens to fit). Deliberately NOT
# replacing res_block1/res_block2/the output head with the transformer directly - user-confirmed
# 2026-08-22: those residual blocks provide non-linear, per-point, regularized (dropout)
# refinement capacity that attention's averaged output doesn't replicate on its own, and replacing
# validated backbone components contradicts this project's "add, don't delete" standing rule.
#
# embed_dim for the late block defaults to 128, not 1024 (unlike the early block, whose embed_dim
# matches its 60-dim input exactly) - deliberately compressed: attention cost scales with embed_dim
# as well as token count, and running full-width 1024-dim attention would partially undo the whole
# point of FPS-downsampling for cost control in the first place.
#
# forward()'s signature matches the other two networks' exactly - a drop-in swap at the
# LightningModule level (models/patch_lightning_module_transformer_deep.py) and at
# PatchCollator.for_network (only reads .k/.dilation_ks, both present unchanged here).


class PatchDilatedToothSegTransformerDeepNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, k=32, dilation_ks=(200, 900, 1800),
                 area_thresholds=(40.0, 180.0, 360.0), dilation_gating=True,
                 transformer_tokens=256, transformer_layers=2, transformer_heads=4,
                 late_transformer_embed_dim=128):
        """Same params as PatchDilatedToothSegTransformerNetwork, plus:
        :param late_transformer_embed_dim: embed_dim for the SECOND (late) GlobalTokenTransformerBlock,
            which sees global_hidden_layer's 1024-dim fused output - compressed down from 1024 by
            default (unlike the early block, which matches its 60-dim input exactly) to keep the
            late attention pass's own cost from partially undoing FPS-downsampling's point.
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

        # EARLY global branch - fed from x (post-local_hidden_layer), same as
        # PatchDilatedToothSegTransformerNetwork's own global_transformer_block.
        self.global_transformer_block = GlobalTokenTransformerBlock(
            in_channels=60, embed_dim=60, num_tokens=transformer_tokens,
            num_layers=transformer_layers, num_heads=transformer_heads)

        self.global_hidden_layer = BasicPointLayer(in_channels=60 * 5, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        # LATE global branch - fed from the gated, fused 1024-dim representation.
        self.late_global_transformer_block = GlobalTokenTransformerBlock(
            in_channels=1024, embed_dim=late_transformer_embed_dim, num_tokens=transformer_tokens,
            num_layers=transformer_layers, num_heads=transformer_heads)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024 + late_transformer_embed_dim,
                                                   out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

    def _resolve_gates(self, area_mm2):
        if not self.dilation_gating or area_mm2 is None:
            return True, True, True
        return tuple(area_mm2 >= t for t in self.area_thresholds)

    def forward(self, x, pos, area_mm2=None, local_idx=None, dilated_idx=(None, None, None), profile=False):
        """Same profiling contract as the other two networks' forward() - see
        PatchDilatedToothSegmentationNetwork's own docstring. Adds two timed stages,
        'early_global_transformer' and 'late_global_transformer'."""
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
        self.last_gates = (gate1, gate2, gate3)
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
        g_early = self.global_transformer_block(x, pos)
        _sync()
        if profile:
            timings['early_global_transformer'] = time.time() - t0

        t0 = time.time()
        x = torch.cat([x, d1, d2, d3, g_early], dim=2)
        x = self.global_hidden_layer(x)
        x = self.feature_importance(x)
        _sync()
        if profile:
            timings['global_hidden_and_importance'] = time.time() - t0

        t0 = time.time()
        g_late = self.late_global_transformer_block(x, pos)
        _sync()
        if profile:
            timings['late_global_transformer'] = time.time() - t0

        t0 = time.time()
        x = torch.cat([x, g_late], dim=2)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        _sync()
        if profile:
            timings['head'] = time.time() - t0
            self.last_timings = timings
        return x
