import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, lens_s, keep_ratio: float,
                          global_index_ps, global_index_x: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_ps + L_s, L_t + L_ps + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_ps + L_s, C], template, ps, and search region tokens
        lens_t (int): length of template
        lens_s (int): length of search and pre_search
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index_ps (torch.Tensor): global index of pre_search region tokens
        global_index_x (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index_ps (torch.Tensor): indices of removed pre_search region tokens
        removed_index_x (torch.Tensor): indices of removed search region tokens
    """
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index_ps, global_index_x, None, None

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_ps = tokens[:, lens_t:lens_t+lens_s]
    tokens_s = tokens[:, lens_t+lens_s:]
    B, L, C = tokens_s.shape

    attn_t = attn[:, :, :lens_t, lens_t:]  # (B, h, L_t, L_ps + L_s)

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s+lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # (B, h, L_t, L_ps + L_s) -> (B, L_ps + L_s)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # (B, h, L_t, L_ps + L_s) -> (B, L_ps + L_s)

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    attn_t_ps, attn_t_s = torch.split(attn_t, [lens_s, lens_s], dim=1)

    sorted_attn, indices = torch.sort(attn_t_ps, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    keep_index_ps = global_index_ps.gather(dim=1, index=topk_idx)
    removed_index_ps = global_index_ps.gather(dim=1, index=non_topk_idx)
    tokens_ps = tokens_ps.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    sorted_attn, indices = torch.sort(attn_t_s, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    keep_index_s = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_s = global_index_x.gather(dim=1, index=non_topk_idx)
    tokens_s = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    # concatenate these tokens
    tokens_new = torch.cat([tokens_t, tokens_ps, tokens_s], dim=1)

    # return tokens_new, keep_index, removed_index
    return tokens_new, keep_index_ps, keep_index_s, removed_index_ps, removed_index_s


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_ps, global_index_search, mask=None,
                ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]
        lens_s = global_index_search.shape[1]

        removed_index_ps = None
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_ps, global_index_search, removed_index_ps, removed_index_search = \
                candidate_elimination(attn, x, lens_t, lens_s, keep_ratio_search, global_index_ps, global_index_search,
                                      ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_ps, global_index_search, removed_index_ps, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
