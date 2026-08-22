from torch import nn

from models.layer import batched_index_select, fps

# New, additive-only file - models/layer.py and models/patch_layer.py are both untouched. Adds
# the one thing genuinely missing from the whole dilated-edge-conv stack (models/patch_layer.py,
# models/patch_dilated_tooth_seg_network.py): true GLOBAL, all-pairs attention. Every existing
# block - local EdgeGraphConvBlock, "dilated" DilatedEdgeGraphConvBlock, PointFeatureImportance -
# is either a fixed-k-nearest-neighbor gather or a per-point-independent channel gate; none of
# them let one point's prediction depend on genuinely ALL of the patch's other points, only on
# progressively wider LOCAL neighborhoods (dilation_ks widens the candidate pool before an FPS
# downsample back to k=32, it never removes the neighborhood-only ceiling). User-confirmed
# 2026-08-22 design, motivated directly by this gap (and by the tooth-4-vs-5 premolar confusion
# this project's own dilation_ks widening experiment already tried, and only partly succeeded, at
# addressing through wider local context alone - see Docs/REALTIME.md).
#
# Full O(N^2) self-attention over every face isn't viable at this project's patch sizes (up to
# 20,000 faces - a (20000,20000) attention matrix per head is already ~1.6GB before even stacking
# layers/heads, on hardware that has already hit GPU driver watchdog kills from a much cheaper
# O(N^2) op - see train_patch_network_color.py's --max_faces comment). Instead:
#   1. FPS-downsample the patch to a small, fixed token budget (fps(), reused unchanged from
#      models/layer.py - same utility DilatedEdgeGraphConvBlock's own fallback path already uses,
#      just called directly on the whole patch instead of per-point-local candidate pools).
#   2. Run standard multi-head self-attention (torch.nn.TransformerEncoder - no new dependency,
#      already part of torch 2.1.0) among just those tokens - tractable at this size, and now
#      genuinely global: every token attends to every other token regardless of distance.
#   3. Broadcast the resulting globally-aware token features back to all N original points via
#      cross-attention (torch.nn.MultiheadAttention, each point as a query, the tokens as
#      keys/values) - O(N x tokens), linear in N, not quadratic. User-chosen 2026-08-22 over a
#      cheaper nearest-neighbor-copy broadcast, for genuine per-point learned weighting over
#      which global tokens matter to it.
#
# No explicit positional encoding added to the token self-attention - deliberately, matching this
# project's own standing preference to avoid raw/absolute-coordinate signals feeding the network
# beyond what's already needed (Docs/MAA_ToDo.md's running-TODO #13). The token FEATURES already
# carry substantial local geometric context by this point in the network (every token is a real
# point that already passed through the local edge-conv stack), which is what's actually attended
# over.
#
# Fed from x (the post-local_hidden_layer features), NOT from any of the 3 dilated blocks'
# outputs - deliberately decoupled from area-gated dilation (models/patch_dilated_tooth_seg_
# transformer_network.py's dilation_gating): x is computed unconditionally on every patch
# regardless of area, so this block's tokens exist and stay meaningful even when all 3 dilated
# gates are off (a small patch) - exactly the situation where it has the most to contribute, since
# d1/d2/d3 all collapse to identity(x) in that case.


class GlobalTokenTransformerBlock(nn.Module):
    """Global-context block: FPS-downsample -> self-attend among tokens -> cross-attend tokens
    back onto every point. Input (B, N, in_channels) + (B, N, 3) positions, output
    (B, N, embed_dim) - an additional feature stream alongside the existing local/dilated blocks,
    not a replacement for them."""

    def __init__(self, in_channels, embed_dim=64, num_tokens=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_proj = nn.Linear(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_proj = nn.Linear(in_channels, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                 dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, pos):
        """x: (B, N, in_channels) point features. pos: (B, N, 3) point positions (same
        patch-centroid-centered, /scale_mm space every other block in this network already uses -
        dataset/patch_preprocessing.py). Returns (B, N, embed_dim)."""
        B, N, _ = x.shape
        n_tokens = min(self.num_tokens, N)  # same small-N clamp pattern as dilation_k/fps_k
        # elsewhere in this codebase (models/patch_layer.py) - a patch smaller than the token
        # budget just uses every point as its own token (full O(N^2) self-attention IS affordable
        # once N is that small).

        token_idx = fps(pos, n_tokens).long()
        token_features = batched_index_select(x, 1, token_idx)

        tokens = self.in_proj(token_features)
        tokens = self.token_encoder(tokens)

        queries = self.query_proj(x)
        point_context, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return self.out_norm(point_context)
