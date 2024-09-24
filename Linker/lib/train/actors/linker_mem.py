from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ...utils.visualize import linker_t_vis
from imgcat import imgcat
import faiss
import torch.distributed as dist
from external.memorizing_transformers_pytorch.memorizing_transformers_pytorch import KNNMemory
# Only calculate loss for the last frame
class LinkerMemActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.faiss = None
        # import ipdb; ipdb.set_trace()
        self.device = dist.get_rank()
        # self.device = 0
        # import ipdb; ipdb.set_trace()
        self.memory = KNNMemory(
                        dim = 768,                   # dimension of key / values
                        # max_memories = 64000,       # maximum number of memories to keep (will throw out the oldest memories for now if it overfills)
                        max_memories = 512,
                        num_indices = 2,             # this should be equivalent to batch dimension, as each batch keeps track of its own memories, expiring when it sees a new document
                        device = self.device
                    )

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out = self.forward_pass(data)

        # compute losses
        if self.cfg.MODEL.SCORE_HEAD:
            loss, status = self.compute_score_loss(out, data)
        else:
            loss, status = self.compute_losses(out, data)

        return loss, status

    def forward_pass(self, data):
        # import ipdb; ipdb.set_trace()
        self.device = data['template_images'].device.index
        # self.memory = KNNMemory(
        #         dim = 768,                   # dimension of key / values
        #         max_memories = 64000,       # maximum number of memories to keep (will throw out the oldest memories for now if it overfills)
        #         num_indices = 4,             # this should be equivalent to batch dimension, as each batch keeps track of its own memories, expiring when it sees a new document
        #         device = self.device
        #     )
        template_img = data['template_images'][0].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
        ps = None
        l_kv = []
        l_k = []
        l_v = []

        out_dict = None

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img.shape[0], template_img.device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        debug = False
        debug_info = {
            'pred_bbox': [],
            'gt_bbox': data['search_anno'].cpu(),
            'template': template_img[0],
            'search': [],
            'pred_score_map': [],
            'pred_score': []
        }
        l_dict = []
        d = 768                           # dimension
        # import ipdb; ipdb.set_trace()
        # cpu_index_k = faiss.IndexFlatL2(d)
        # gpu_index_k = faiss.index_cpu_to_all_gpus(cpu_index_k)
        # cpu_index_v = faiss.IndexFlatL2(d)
        # gpu_index_v = faiss.index_cpu_to_all_gpus(cpu_index_v)
        # index_k = faiss.IndexFlatL2(768)
        # index_v = faiss.IndexFlatL2(64)
        # l_k = []
        # l_v = []
        for search in data['search_images']:
            search_img = search.view(-1, *search.shape[1:])  # (batch, 3, 320, 320)
            out_dict = self.net(template=template_img,
                                l_kv=l_kv,
                                # memory=[gpu_index_k, gpu_index_v],
                                memory = self.memory,
                                search=search_img,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                debug=debug)
            # ps = out_dict['s_feat']
            # import ipdb; ipdb.set_trace()
            l_kv.append('1')
            # index_k.add(out_dict['l_kv'][0]) 
            # import ipdb; ipdb.set_trace()
            self.memory.add(out_dict['l_kv'])

            # kk, vv = torch.unbind(out_dict['l_kv'][0],1)
            # kkk = kk.cpu().detach().numpy()
            # vvv = vv.cpu().detach().numpy()
            # l_kkk
            # l_vvv
            # gpu_index_k.add(kkk)
            # gpu_index_v.add(vvv)
            # k = 4                          # we want to see 4 nearest neighbors
            # D, I = gpu_index.search(q, k) # actual search 
            # add = out_dict['l_kv'].squeeze(1)
            # memory.add(add)
            # l_kv = out_dict['l_kv']
            # l_k.append(out_dict['l_kv'][0])

            # l_kv = index_k(search(l_kv))
            if debug:
                debug_info['pred_bbox'].append(out_dict['pred_boxes'][0, 0].tolist())
                debug_info['search'].append(search[0])
                debug_info['pred_score_map'].append(out_dict['score_map'][0, 0].detach().cpu())
                debug_info['pred_score'].append(1 if 'pred_score' not in out_dict else torch.sigmoid(out_dict['pred_score'][0]))
            l_dict.append(out_dict)
        self.memory.clear()
        # self.memory = None
        if debug:
            linker_t_vis(debug_info)
        del debug_info

        # return out_dict
        return l_dict

    def compute_score_loss(self, pred_dict, gt_dict):
        pred_boxes = pred_dict['pred_boxes']
        gt_boxes = gt_dict['search_anno'][-1]  # (B, 4)
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (BN, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_boxes)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                            max=1.0)
        iou, _ = box_iou(pred_boxes_vec, gt_boxes_vec)

        pred_score = pred_dict['pred_score'].view(-1)
        gt_score = gt_dict['search_visible'].bool() & (iou > 0.5)
        gt_score = gt_score.float().view(-1)

        loss = self.objective['cls'](pred_score, gt_score)
        status = {'Loss/score': loss.item()}
        return loss, status

    def compute_losses(self, l_pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        loss = 0
        for i, pred_dict in enumerate(l_pred_dict):
            # import ipdb; ipdb.set_trace()
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'][i].unsqueeze(0), self.cfg.DATA.SEARCH.SIZE,
                                                self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                            max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            # weighted sum
            loss += self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
                'focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

# class LinkerTActor(BaseActor):
#     """ Actor for training OSTrack models """
#
#     def __init__(self, net, objective, loss_weight, settings, cfg=None):
#         super().__init__(net, objective)
#         self.loss_weight = loss_weight
#         self.settings = settings
#         self.bs = self.settings.batchsize  # batch size
#         self.cfg = cfg
#
#     def __call__(self, data):
#         """
#         args:
#             data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
#             template_images: (N_t, batch, 3, H, W)
#             search_images: (N_s, batch, 3, H, W)
#         returns:
#             loss    - the training loss
#             status  -  dict containing detailed losses
#         """
#         # forward pass
#         out = self.forward_pass(data)
#
#         # compute losses
#         loss, status = self.compute_losses(out, data['search_anno'])
#
#         return loss, status
#
#     def forward_pass(self, data):
#         template_img = data['template_images'][0].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
#         ps = None
#         out = []
#
#         box_mask_z = None
#         ce_keep_rate = None
#         if self.cfg.MODEL.BACKBONE.CE_LOC:
#             box_mask_z = generate_mask_cond(self.cfg, template_img.shape[0], template_img.device,
#                                             data['template_anno'][0])
#
#             ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
#             ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
#             ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
#                                             total_epochs=ce_start_epoch + ce_warm_epoch,
#                                             ITERS_PER_EPOCH=1,
#                                             base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
#
#         debug = False
#         for search in data['search_images']:
#             search_img = search.view(-1, *search.shape[1:])  # (batch, 3, 320, 320)
#             out_dict, debug_info = self.net(template=template_img,
#                                             ps=ps,
#                                             search=search_img,
#                                             ce_template_mask=box_mask_z,
#                                             ce_keep_rate=ce_keep_rate,
#                                             return_last_attn=False,
#                                             debug=debug)
#             ps = out_dict['s_feat'].detach()
#             out.append(out_dict)
#
#         # if debug:
#         #     gt_bbox = data['search_anno'][-1][0].cpu()
#         #     x1, y1, w, h = gt_bbox
#         #     gt_bbox = [x1 + w / 2, y1 + h / 2, w, h]
#         #     linker_vis(debug_info, gt_bbox=gt_bbox)
#         # del debug_info
#
#         return out
#
#     def compute_losses(self, preds, gts, return_status=True):
#         mean_giou_loss = 0
#         mean_l1_loss = 0
#         mean_focal_loss = 0
#         mean_iou = 0
#
#         for pred_dict, gt_bbox in zip(preds, gts):
#             # gt gaussian map
#             gt_gaussian_maps = generate_heatmap([gt_bbox], self.cfg.DATA.SEARCH.SIZE,
#                                                 self.cfg.MODEL.BACKBONE.STRIDE)
#             gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
#
#             # Get boxes
#             pred_boxes = pred_dict['pred_boxes']
#             if torch.isnan(pred_boxes).any():
#                 raise ValueError("Network outputs is NAN! Stop Training")
#             num_queries = pred_boxes.size(1)
#             pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
#             gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
#                                                                                                                max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
#             # compute giou and iou
#             try:
#                 giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
#             except:
#                 giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
#             # compute l1 loss
#             l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
#             # compute location loss
#             if 'score_map' in pred_dict:
#                 location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
#             else:
#                 location_loss = torch.tensor(0.0, device=l1_loss.device)
#
#             mean_giou_loss += giou_loss
#             mean_l1_loss += l1_loss
#             mean_focal_loss += location_loss
#             mean_iou += iou.detach().mean()
#
#         mean_giou_loss /= len(preds)
#         mean_l1_loss /= len(preds)
#         mean_focal_loss /= len(preds)
#         mean_iou /= len(preds)
#         loss = mean_giou_loss * self.loss_weight['giou'] + mean_l1_loss * self.loss_weight['l1'] + \
#                mean_focal_loss * self.loss_weight['focal']
#
#         if return_status:
#             # status for log
#             status = {"Loss/total": loss.item(),
#                       "Loss/giou": mean_giou_loss.item(),
#                       "Loss/l1": mean_l1_loss.item(),
#                       "Loss/location": mean_focal_loss.item(),
#                       "IoU": mean_iou.item()}
#             return loss, status
#         else:
#             return loss
