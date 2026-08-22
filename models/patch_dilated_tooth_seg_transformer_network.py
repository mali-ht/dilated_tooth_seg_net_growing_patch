import time

import torch
from torch import nn

from models.patch_layer import (BasicPointLayer, DilatedEdgeGraphConvBlock, EdgeGraphConvBlock,
                                 PointFeatureImportance, ResidualBasicPointLayer)
from models.patch_transformer_layer import GlobalTokenTransformerBlock

# New, additive-only file - models/patch_dilated_tooth_seg_network.py (Component 4, itself already
# a from-scratch-forward surgery of the protected original) is untouched. Same backbone (local
# edge-conv x3 -> dilated edge-conv x3 -> global head) with ONE addition: a
# GlobalTokenTransformerBlock (models/patch_transformer_layer.py) run in parallel with the 3
# dilated blocks, giving every point genuine all-pairs global context - something no existing
# block in this architecture provides (see that file's own module docstring for the full
# reasoning). Its output is concatenated alongside x/d1/d2/d3 into global_hidden_layer, which
# gains one more block width of input channels (60*5 instead of 60*4) to match - everything else
# (local blocks, dilated blocks, head) is copy-identical to PatchDilatedToothSegmentationNetwork,
# not reimplemented differently, so a head-to-head comparison isolates just this one addition.
#
# forward()'s signature matches PatchDilatedToothSegmentationNetwork's exactly (same
# local_idx/dilated_idx/area_mm2/profile args) - a drop-in swap at the LightningModule level (see
# models/patch_lightning_module_transformer.py) and at PatchCollator.for_network (only reads
# .k/.dilation_ks, both present unchanged here), not a differently-shaped model needing its own
# training/collate machinery.


class PatchDilatedToothSegTransformerNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, k=32, dilation_ks=(200, 900, 1800),
                 area_thresholds=(40.0, 180.0, 360.0), dilation_gating=True,
                 transformer_tokens=256, transformer_layers=2, transformer_heads=4):
        """Same params as PatchDilatedToothSegmentationNetwork, plus:
        :param transformer_tokens: FPS token budget for GlobalTokenTransformerBlock (clamped to N
            for small patches).
        :param transformer_layers: Number of self-attention layers among those tokens.
        :param transformer_heads: Attention heads, both for token self-attention and the
            point<-tokens cross-attention broadcast.
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

        # Fed from x (post-local_hidden_layer), not from d1/d2/d3 - deliberately decoupled from
        # dilation_gating, see this file's own module docstring and
        # models/patch_transformer_layer.py's.
        self.global_transformer_block = GlobalTokenTransformerBlock(
            in_channels=60, embed_dim=60, num_tokens=transformer_tokens,
            num_layers=transformer_layers, num_heads=transformer_heads)

        self.global_hidden_layer = BasicPointLayer(in_channels=60 * 5, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

    def _resolve_gates(self, area_mm2):
        if not self.dilation_gating or area_mm2 is None:
            return True, True, True
        return tuple(area_mm2 >= t for t in self.area_thresholds)

    def forward(self, x, pos, area_mm2=None, local_idx=None, dilated_idx=(None, None, None), profile=False):
        """Same profiling contract as PatchDilatedToothSegmentationNetwork.forward() - see that
        file's own docstring. Adds one more timed stage, 'global_transformer', between the
        dilated blocks and the head."""
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
        g = self.global_transformer_block(x, pos)
        _sync()
        if profile:
            timings['global_transformer'] = time.time() - t0

        t0 = time.time()
        x = torch.cat([x, d1, d2, d3, g], dim=2)
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
