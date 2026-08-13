import torch
from torch import nn

from models.layer import batched_index_select, fps

# Component 4 (claude_code_prompt.md) model surgery, adapted from models/layer.py (untouched -
# see dataset/patch_*.py for the same "new file, original stays as-is" pattern used throughout
# this project). Two independent changes from the original:
#
# 1. BatchNorm -> GroupNorm everywhere a norm layer appears (_group_norm below), so normalization
#    is per-sample and independent of batch/patch composition - see TRAINING_CONCERNS.md and
#    claude_code_prompt.md Component 4. GroupNorm behaves identically in train/eval mode (no
#    running statistics), which matters here specifically because our patches vary wildly in
#    both size and composition (a lone incisor vs. a near-full-arch sweep) - a BatchNorm running
#    average accumulated across that mix doesn't represent any single patch well, unlike the
#    original paper's fixed-16,000-face, roughly-constant-composition setting.
#
# 2. A hard N<=32 crash, confirmed empirically (see TRAINING_CONCERNS.md #1): EdgeGraphConvBlock's
#    knn() calls topk(k=33), which requires >=33 points to exist at all. Our early_bias_power
#    sampling deliberately produces patches far smaller than that. Fixed by clamping k to N-1
#    wherever a fixed k is used (knn() below, and DilatedEdgeGraphConvBlock's on-the-fly fallback
#    path) - mirroring the dilation_k = min(dilation_k, N) guard the ORIGINAL code already has for
#    the dilated blocks, just extended to the local blocks which had no equivalent guard.
#
# Both edge-conv blocks also accept fully precomputed neighbor indices (idx / idx_fps), bypassing
# their internal cdist/topk/FPS entirely when supplied - see models/patch_collate.py, which
# precomputes these on CPU wherever the neighbor graph is purely a function of metric position
# (fixed before the network runs), leaving only the two truly-dynamic feature-space kNN lookups
# (in the 2nd/3rd local blocks) to run on-the-fly. batched_index_select and fps are reused
# unchanged from models.layer (pure utility functions, read-only import - not a modification).


def _group_norm(num_channels, max_groups=8):
    """GroupNorm with the largest group count <= max_groups that evenly divides num_channels
    (falls back to 1, i.e. LayerNorm-equivalent, in the worst case) - avoids hardcoding a group
    count that happens to work for this architecture's specific channel sizes (24, 60, ...) but
    would silently become invalid if those change."""
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


def knn(x, k=16):
    """Same as models.layer.knn, but clamps k to N-1 (the most neighbors that can possibly exist)
    instead of crashing when N <= k. Returns a (batch_size, num_points, k_effective) index tensor
    - k_effective may be smaller than the requested k for small patches, which callers must not
    assume is fixed (see get_graph_feature's k = idx.size(2) below)."""
    x_t = x.transpose(2, 1)
    n = x_t.size(1)
    k_eff = max(0, min(k, n - 1))
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k_eff + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def get_graph_feature(x, k=20, idx=None, pos=None, edge_function='global'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    # trust idx's own width, not the requested k - it may have been clamped (small-N) or be an
    # externally precomputed tensor with a different width than whatever k was passed in here
    k = idx.size(2)
    device = x.device

    idx_org = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if edge_function == 'global':
        feature = feature.permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local':
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local_global':
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)


class EdgeGraphConvBlock(nn.Module):
    """Same as models.layer.EdgeGraphConvBlock, with GroupNorm instead of BatchNorm2d. The
    idx-bypass and small-N clamp already flow through from get_graph_feature/knn above, so no
    other logic changes are needed here."""

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, k=32):
        super().__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function == "local_global":
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos=None, idx=None):
        x_t = x.transpose(2, 1)
        if pos is None:
            pos = x
        pos_t = pos.transpose(2, 1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx, pos=pos_t)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class DilatedEdgeGraphConvBlock(nn.Module):
    """
    Same as models.layer.DilatedEdgeGraphConvBlock, with GroupNorm instead of BatchNorm2d, plus a
    new idx_fps parameter: a fully precomputed, already-FPS-resolved (B, N, k) neighbor index
    tensor. When supplied, cdist/topk/FPS are skipped entirely - this is the fast path, meant to
    be fed by models/patch_collate.py's CPU precompute (the dilated neighborhood is a function of
    metric position alone, fixed before the network runs, so it never needs to happen on GPU at
    all - see that file's docstring).

    Without idx_fps, falls back to the original on-the-fly computation, now with the same small-N
    clamp applied to the FPS sample count as to dilation_k itself: the original code already did
    dilation_k = min(dilation_k, N) but did not also clamp self.k for the subsequent FPS call, so
    a patch smaller than k=32 faces would ask FPS to sample more points than its candidate pool
    contains. This fallback path exists for standalone testability - real training/inference uses
    the precomputed path.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, dilation_k=128, k=32):
        super().__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if dilation_k < k:
            raise ValueError(f'Dilation k {dilation_k} must be larger than k {k}')
        self.dilation_k = dilation_k
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function in ["local_global"]:
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos=None, cd=None, idx_fps=None):
        x_t = x.transpose(2, 1)
        B, N, C = x.shape
        if idx_fps is None:
            if pos is None:
                raise ValueError('DilatedEdgeGraphConvBlock needs either idx_fps or pos (to compute it)')
            if cd is None:
                cd = torch.cdist(pos, pos, p=2)
            dilation_k = min(self.dilation_k, N)
            fps_k = min(self.k, dilation_k)
            idx_l = torch.topk(cd, dilation_k, largest=False)[1].reshape(B * N, -1)
            idx_sampled = fps(pos.reshape(B * N, -1)[idx_l], fps_k).long()
            idx_fps = batched_index_select(idx_l, 1, idx_sampled).reshape(B, N, -1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx_fps)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class BasicPointLayer(nn.Module):
    """Same as models.layer.BasicPointLayer, with GroupNorm instead of BatchNorm1d."""

    def __init__(self, in_channels, out_channels, dropout=0.1, is_out=False):
        super().__init__()
        if is_out:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                _group_norm(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1)


class ResidualBasicPointLayer(nn.Module):
    """Same as models.layer.ResidualBasicPointLayer, with GroupNorm instead of BatchNorm1d."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        if in_channels != out_channels:
            self.rescale = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                _group_norm(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)


class PointFeatureImportance(nn.Module):
    """Same as models.layer.PointFeatureImportance, with GroupNorm instead of BatchNorm1d."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            _group_norm(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)
