"""
Basic OSTrack model.
"""
import math
import os
from collections import OrderedDict
from functools import partial
from typing import List

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.transformer import _get_clones

from external.UniFormer.video_classification.slowfast.utils import logging
from lib.models.layers.head import build_box_head, ScoreHead
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from external.UniFormer.video_classification.slowfast.models.uniformer import SpeicalPatchEmbed, PatchEmbed, \
    CBlock, SplitSABlock, SABlock, bn_3d, model_path


logger = logging.get_logger(__name__)


class Uniformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self):
        super().__init__()

        depth = [5, 8, 20, 7]
        img_size = 256
        in_chans = 3
        embed_dim = [64, 128, 320, 512]
        head_dim = 64
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        representation_size = None
        drop_rate = 0
        attn_drop_rate = 0
        drop_path_rate = 0.1
        split = False
        std = False
        self.use_checkpoint = False
        self.checkpoint_num = [0, 0, 0, 0]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
                for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
                for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, pretrain_name=None):
        if pretrain_name:
            checkpoint = torch.load(pretrain_name, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        logger.info(f'Ignore: {k}')
                        continue
                    logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed2(x)
        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed3(x)
        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed4(x)
        for i, blk in enumerate(self.blocks4):
            if self.use_checkpoint and i < self.checkpoint_num[3]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = x[0]
        x = self.forward_features(x)
        return x


class LinkerUniFormer(nn.Module):
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
                debug=False
                ):
        x = torch.cat((template, search), dim=2)  # (B, C, T, H, W)
        feat = self.backbone.forward_features(x)

        # Forward head
        feat_last = feat[:, :, -1]  # (B, C, H, W)
        out = self.forward_head(feat_last, None)

        # if self.score_head is not None:
        #     score_map = out['score_map']
        #     B, _, H, W = score_map.shape
        #     s = rearrange(s, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        #     score_input = torch.cat((s.detach(), score_map.detach()), dim=1)
        #     out.update({'pred_score': self.score_head(score_input).view(-1)})

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


def build_linker_uniformer(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models/uniformer_base_in1k.pth')
    backbone = Uniformer()
    backbone.load_state_dict(backbone.get_pretrained_model(pretrained_path), strict=False)
    logger.info('load pretrained model')
    hidden_dim = backbone.embed_dim[-1]

    box_head = build_box_head(cfg, hidden_dim)
    if cfg.MODEL.SCORE_HEAD:
        score_head = ScoreHead(
            in_planes=hidden_dim + 1,
            channel=cfg.MODEL.HEAD.NUM_CHANNELS,
            mlp_in=(cfg.DATA.SEARCH.SIZE // 16) ** 2,
            freeze_bn=False
        )
        model = LinkerUniFormer(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
            score_head=score_head
        )
    else:
        model = LinkerUniFormer(
            backbone,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
        )

    if 'LinkerUniFormer' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
