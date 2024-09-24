"""
Basic OSTrack model.
"""
import math
import os
import logging

import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.transformer import _get_clones
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_, DropPath, Mlp
from timm.models.helpers import named_apply

from lib.models.layers.head import build_box_head, ScoreHead
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.patch_embed import PatchEmbed
from .utils import recover_tokens
from .vit import Block, _init_vit_weights, _load_weights
from ..layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def attn(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'b h t d -> b t (h d)')
        return x, attn

    def forward(self, x, lens_ts, lens_m, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool

        read_memory, ts, write_memory = torch.split(x, [lens_m, lens_ts, lens_m], dim=1)

        B, N, C = read_memory.shape
        qkv_r = self.qkv(read_memory).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, h, N, C//h)
        q_r, k_r, v_r = qkv_r.unbind(0)

        B, N, C = ts.shape
        qkv_ts = self.qkv(ts).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, h, N, C//h)
        q_ts, k_ts, v_ts = qkv_ts.unbind(0)

        B, N, C = write_memory.shape
        qkv_w = self.qkv(write_memory).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, h, N, C//h)
        q_w, k_w, v_w = qkv_w.unbind(0)

        read_memory, _ = self.attn(q_r, k_r, v_r)
        ts, attn = self.attn(q_ts, torch.cat([k_r, k_ts], dim=2), torch.cat([v_r, v_ts], dim=2))
        write_memory, _ = self.attn(q_w, torch.cat([k_r, k_ts, k_w], dim=2), torch.cat([v_r, v_ts, v_w], dim=2))
        x = torch.cat([read_memory, ts, write_memory], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            attn_ts = attn[:, :, :, lens_m:]
            return x, attn_ts
        return x


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, lens_s, lens_m, keep_ratio: float,
                          global_index_x: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_ps + L_s, L_t + L_ps + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_ps + L_s, C], template, ps, and search region tokens
        lens_t (int): length of template
        lens_s (int): length of search and pre_search
        keep_ratio (float): keep ratio of search region tokens (candidates)
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
        return tokens, global_index_x, None

    # separate template and search tokens
    tokens_read, tokens_t, tokens_s, tokens_write = torch.split(tokens, [lens_m, lens_t, lens_s, lens_m], dim=1)
    B, L, C = tokens_s.shape

    attn_t = attn[:, :, :lens_t, lens_t:lens_t+lens_s]  # (B, h, L_t, L_s)

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # (B, h, L_t, L_s) -> (B, L_s)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # (B, h, L_t, L_s) -> (B, L_s)

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    keep_index_s = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_s = global_index_x.gather(dim=1, index=non_topk_idx)
    tokens_s = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    # concatenate these tokens
    tokens_new = torch.cat([tokens_read, tokens_t, tokens_s, tokens_write], dim=1)

    # return tokens_new, keep_index, removed_index
    return tokens_new, keep_index_s, removed_index_s


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None,
                ce_template_mask=None, keep_ratio_search=None):
        lens_t = global_index_template.shape[1]
        lens_s = global_index_search.shape[1]
        lens_m = (x.shape[1] - lens_t - lens_s) // 2
        x_attn, attn = self.attn(self.norm1(x), lens_ts=lens_t+lens_s, lens_m=lens_m, return_attention=True)
        x = x + self.drop_path(x_attn)

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = \
                candidate_elimination(attn, x, lens_t, lens_s, lens_m, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_t = None
        self.pos_embed_s = None
        self.pos_embed_read = None
        self.pos_embed_write = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE
        memory_size = cfg.MODEL.MEMORY_SIZE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE  # direct
        self.return_inter = cfg.MODEL.RETURN_INTER  # False
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG  # False

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed_s = nn.Parameter(search_patch_pos_embed)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed_t = nn.Parameter(template_patch_pos_embed)

        # for read & write memory
        # H = W = int(math.sqrt(memory_size))
        # assert H * W == memory_size
        # memory_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        # read_pos_embed = memory_pos_embed.flatten(2).transpose(1, 2)
        # write_pos_embed = read_pos_embed.clone()
        # self.pos_embed_read = nn.Parameter(read_pos_embed)
        # self.pos_embed_write = nn.Parameter(write_pos_embed)

        self.pos_embed_read = nn.Parameter(torch.zeros(1, memory_size, self.embed_dim))
        trunc_normal_(self.pos_embed_read, std=.02)
        self.pos_embed_write = nn.Parameter(torch.zeros(1, memory_size, self.embed_dim))
        trunc_normal_(self.pos_embed_write, std=.02)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    # def forward_features(self, t, s, m):
    #     B, H, W = s.shape[0], s.shape[2], s.shape[3]
    #
    #     t = self.patch_embed(t)
    #     s = self.patch_embed(s)
    #     m = self.patch_embed(m)
    #
    #     if self.add_cls_token:
    #         cls_tokens = self.cls_token.expand(B, -1, -1)
    #         cls_tokens = cls_tokens + self.cls_pos_embed
    #
    #     t += self.pos_embed_t
    #     m += self.pos_embed_m
    #     s += self.pos_embed_s
    #
    #     if self.add_sep_seg:
    #         s += self.search_segment_pos_embed
    #         t += self.template_segment_pos_embed
    #
    #     # x = combine_tokens(z, x, mode=self.cat_mode)
    #     x = torch.cat((t, m, s), dim=1)
    #     if self.add_cls_token:
    #         x = torch.cat([cls_tokens, x], dim=1)
    #
    #     x = self.pos_drop(x)
    #
    #     for i, blk in enumerate(self.blocks):
    #         x = blk(x)
    #
    #     lens_t = self.pos_embed_t.shape[1]
    #     lens_s = self.pos_embed_s.shape[1]
    #     lens_m = self.pos_embed_m.shape[1]
    #     x = recover_tokens(x, lens_t, lens_m, lens_s, mode=self.cat_mode)
    #
    #     aux_dict = {"attn": None}
    #     return self.norm(s), aux_dict
    #
    # def forward(self, t, s, m, **kwargs):
    #     """
    #     Joint feature extraction and relation modeling for the basic ViT backbone.
    #     Args:
    #         t (torch.Tensor): template feature, [B, C, H_z, W_z]
    #         s (torch.Tensor): search region feature, [B, C, H_x, W_x]
    #         m (torch.Tensor): memory feature, [B, C, L]
    #
    #     Returns:
    #         x (torch.Tensor): merged template and search region feature, [B, L_z+L_x+L_x, C]
    #         attn : None
    #     """
    #     x, aux_dict = self.forward_features(t, s, m)
    #
    #     return x, aux_dict


class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # # Representation layer
        # if representation_size and not distilled:
        #     self.num_features = representation_size
        #     self.pre_logits = nn.Sequential(OrderedDict([
        #         ('fc', nn.Linear(embed_dim, representation_size)),
        #         ('act', nn.Tanh())
        #     ]))
        # else:
        #     self.pre_logits = nn.Identity()
        #
        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, temporal_mode='rnn'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.temporal_mode = temporal_mode
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, t, s, m, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = s.shape[0], s.shape[2], s.shape[3]
        lens_t = self.pos_embed_t.shape[1]
        lens_s = self.pos_embed_s.shape[1]
        lens_m = self.pos_embed_read.shape[1]

        t = self.patch_embed(t)
        t = t + self.pos_embed_t
        s = self.patch_embed(s)
        s = s + self.pos_embed_s
        if m is None:
            m = torch.zeros((B, lens_m, self.embed_dim), device=t.device)
        read_memory = m + self.pos_embed_read
        write_memory = m + self.pos_embed_write
        x = torch.cat((read_memory, t, s, write_memory), dim=1)
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_t - 1, lens_t).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_s - 1, lens_s).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []

        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_t_new = global_index_t.shape[1]
        lens_s_new = global_index_s.shape[1]

        read_memory, t, s, write_memory = torch.split(x, (lens_m, lens_t_new, lens_s_new, lens_m), dim=1)

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_s = lens_s - lens_s_new
            pad_s = torch.zeros([B, pruned_lens_s, s.shape[2]], device=s.device)
            s = torch.cat([s, pad_s], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = s.shape[-1]
            s = torch.zeros_like(s).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=s)
        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([t, s], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "memory": write_memory
        }

        return x, aux_dict

    def forward(self, t, s, m, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        return self.forward_features(t, s, m, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


class LinkerM(nn.Module):
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", score_head=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.score_head = score_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self,
                template: torch.Tensor,
                search: torch.Tensor,
                memory: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                debug=False
                ):
        x, aux_dict = self.backbone(t=template, s=search, m=memory,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x

        if self.score_head is not None:
            score_map = out['score_map']
            B, _, H, W = score_map.shape
            s = rearrange(x[:, -self.feat_len_s:], 'b (h w) c -> b c h w', h=H, w=W).contiguous()
            score_input = torch.cat((s.detach(), score_map.detach()), dim=1)
            out.update({'pred_score': self.score_head(score_input).view(-1)})

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_linker_m(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('LinkerM' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    if cfg.MODEL.SCORE_HEAD:
        score_head = ScoreHead(
            in_planes=hidden_dim + 1,
            channel=cfg.MODEL.HEAD.NUM_CHANNELS,
            mlp_in=(cfg.DATA.SEARCH.SIZE // 16) ** 2,
            freeze_bn=False
        )
        model = LinkerM(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
            score_head=score_head
        )
    else:
        model = LinkerM(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
        )

    if 'LinkerM' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
