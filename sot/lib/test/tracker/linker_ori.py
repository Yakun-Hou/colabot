import math
import shutil

from lib.models.ostrack import build_linker
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_xywh_to_cxcywh
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.visualize import linker_vis


class Linker(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        network = build_linker(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = False#params.debug
        self.vis_layers = [3, 6, 9, 11]  # visualize the 1, 4, 7, 10, 12 layer's attention
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug/%s" % dataset_name
                # if os.path.exists(self.save_dir):
                #     shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)
            else:
                # self.add_hook()
                self._init_visdom(visdom_info={'server': 'http://cluster-proxy.sh.sensetime.com', 'port': 20533}, debug=1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                   output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        self.template = self.preprocessor.process(z_patch_arr)

        ps_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.search_factor,
                                        output_sz=self.params.search_size)
        self.ps_patch_arr = ps_patch_arr
        self.ps = self.preprocessor.process(ps_patch_arr)

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        self.template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, self.template.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.ps_idx = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, search_bbox = sample_target(image, self.state,
                                                                self.params.search_factor,
                                                                output_sz=self.params.search_size,
                                                                return_crop_bbox=True)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            # visualize attention weight
            if self.debug:
                attn_weights = []
                hooks = []
                for i in self.vis_layers:  # layer_idx
                    hooks.append(
                        self.network.backbone.blocks[i].attn.attn_drop.register_forward_hook(
                            lambda self, input, output: attn_weights.append(output[0])
                        ))

            out_dict, debug_info = self.network.forward(self.template, self.ps, search,
                                                        ce_template_mask=self.box_mask_z,
                                                        debug=self.debug)

        # add hann windows
        pred_score_map = out_dict['score_map']
        # response = self.output_window * pred_score_map
        response = pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            print('#%04d' % self.frame_id)
            if not self.use_visdom:
                # # simple visualization
                # x1, y1, w, h = self.state
                # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                # x1, y1, w, h = info['gt_bbox'].tolist()
                # if not math.isnan(x1):
                #     cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=2)
                # x1, y1, w, h = search_bbox
                # cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(255, 0, 0), thickness=2)
                # cv2.putText(image_BGR, '#%04d' % self.frame_id, (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(image_BGR, '%.2f' % float(max_score), (10, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                # cv2.imwrite(save_path, image_BGR)

                # detailed visualization
                for hook in hooks:
                    hook.remove()
                os.makedirs(os.path.join(self.save_dir, info['seq_name']), exist_ok=True)
                save_path = os.path.join(self.save_dir, info['seq_name'], "%04d.jpg" % self.frame_id)
                # save_path = None
                x1, y1, w, h = info['gt_bbox'].tolist()
                s_x1, s_y1, s_w, s_h = search_bbox
                x1 = (x1 - s_x1) / s_w
                y1 = (y1 - s_y1) / s_h
                w /= s_w
                h /= s_h
                if not math.isnan(x1):
                    gt_bbox = [x1 + w / 2, y1 + h / 2, w, h]
                else:
                    gt_bbox = None
                linker_vis(debug_info, self.ps_idx, self.frame_id, gt_bbox=gt_bbox, save_path=save_path, attn_weights=attn_weights, vis_layers=self.vis_layers)
            else:
                cv2.putText(image, '#%03d' % self.frame_id, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(image, '%.2f' % float(max_score), (10, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.ps_patch_arr).permute(2, 0, 1), 'image', 1, 'ps')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        # update ps
        max_score = pred_score_map.max()
        pred_score = torch.sigmoid(out_dict['pred_score']) if self.cfg.MODEL.SCORE_HEAD else 1
        update_ps = max_score >= 0.6 and pred_score >= 0.6
        if update_ps:
            ps_patch_arr, _ = sample_target(image, self.state, self.params.search_factor,
                                            output_sz=self.params.search_size)
            self.ps_patch_arr = ps_patch_arr
            self.ps = self.preprocessor.process(ps_patch_arr)
            self.ps_idx = self.frame_id

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return Linker
