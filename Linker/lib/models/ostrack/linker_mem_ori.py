"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, ScoreHead
from lib.models.ostrack.vit import vit_base_patch16_224
# from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
# from ..layers.attn_blocks import CEBlock
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


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

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        split_attn = False
        len_t = 49
        if split_attn:
            attn_t = attn[..., :len_t].softmax(dim=-1)
            attn_s = attn[..., len_t:].softmax(dim=-1)
            attn = torch.cat([attn_t, attn_s], dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn, [k, v]
        else:
            return x, [k, v]

class Linker(nn.Module):
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
                ps: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                debug=False
                ):
        x, aux_dict, s = self.backbone(z=template,
                                       ps=ps,
                                       x=search,
                                       ce_template_mask=ce_template_mask,
                                       ce_keep_rate=ce_keep_rate,
                                       return_last_attn=return_last_attn,
                                       debug=debug)

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
            s = rearrange(s, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
            score_input = torch.cat((s.detach(), score_map.detach()), dim=1)
            out.update({'pred_score': self.score_head(score_input).view(-1)})

        if debug:
            debug_info = {
                'template': template[0],
                'ps': ps[0],
                'search': search[0],
                'pred_bbox': out['pred_boxes'][0, 0].detach().cpu(),
                'score_map': None if 'score_map' not in out else out['score_map'][0, 0].detach().cpu(),  # (H, W)
                'pred_score': None if 'pred_score' not in out else out['pred_score'][0],  # (1)
                'l_s_patch_ids': out['l_s_patch_ids'],  # L x (Ns)
                'l_ps_patch_ids': out['l_ps_patch_ids']
            }
        else:
            debug_info = {}

        return out, debug_info

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


def build_linker_mem(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('Linker' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
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
        model = Linker(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
            score_head=score_head
        )
    else:
        model = Linker(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
        )

    if 'Linker' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


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
                 ce_loc=None, ce_keep_ratio=None):
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
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
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

    def forward_features(self, z, ps, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, debug=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if ps is None:
            ps = x
        x = self.patch_embed(x)
        ps = self.patch_embed(ps)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        ps += self.pos_embed_ps
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        # x = combine_tokens(z, x, mode=self.cat_mode)
        x = torch.cat((z, ps, x), dim=1)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_ps = self.pos_embed_ps.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_ps = torch.linspace(0, lens_ps - 1, lens_ps).to(x.device)
        global_index_ps = global_index_ps.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_ps = []
        removed_indexes_s = []
        l_s_patch_ids = []
        l_ps_patch_ids = []
        l_kv = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_ps, global_index_s, removed_index_ps, removed_index_s, attn, kv = \
                blk(x, global_index_t, global_index_ps, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            l_kv.append(kv)
            if debug:
                l_s_patch_ids.append(global_index_s[0].clone().long())  # (n_s)
                l_ps_patch_ids.append(global_index_ps[0].clone().long())

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_ps.append(removed_index_ps)
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_ps_new = global_index_ps.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        ps = x[:, lens_z_new:lens_z_new+lens_ps_new]
        x = x[:, lens_z_new+lens_ps_new:]

        if removed_indexes_ps and removed_indexes_ps[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_ps, dim=1)

            pruned_lens_ps = lens_ps - lens_ps_new
            pad_ps = torch.zeros([B, pruned_lens_ps, ps.shape[2]], device=ps.device)
            ps = torch.cat([ps, pad_ps], dim=1)
            index_all = torch.cat([global_index_ps, removed_indexes_cat], dim=1)
            # recover original token order
            C = ps.shape[-1]
            ps = torch.zeros_like(ps).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=ps)

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        # re-concatenate with the template, which may be further used by other modules
        feat = torch.cat([z, ps, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "l_s_patch_ids": l_s_patch_ids,
            "l_ps_patch_ids": l_ps_patch_ids
        }

        return feat, aux_dict, x

    def forward(self, z, ps, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False, debug=False):
        x, aux_dict, s = self.forward_features(z, ps, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, debug=debug)

        return x, aux_dict, s


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


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

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
        x_attn, attn, l_kv = self.attn(self.norm1(x), mask, True)
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
        return x, global_index_template, global_index_ps, global_index_search, removed_index_ps, removed_index_search, attn, l_kv


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
