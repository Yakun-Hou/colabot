

import os
from running import TensorRTTracker

from _collections import OrderedDict
import math
import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import random

SSIZE=288
TSIZE=128
init_iou=0.5
init_a=1
min_score=0.5

def _read_image(path):
    print(path)
    im = cv.imread(path, cv.IMREAD_COLOR)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32)
def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]

class Preprocessor_wo_mask(object):
    def __init__(self,cfg=None):
        if cfg==None:
            self.mean=torch.tensor([0.426,0.430,0.420]).view((1, 3, 1, 1)).cuda()
            self.std=torch.tensor([0.196,0.198,0.212]).view((1, 3, 1, 1)).cuda()
        else:
            self.mean=torch.tensor(cfg.DATA.MEAN).view((1, 3, 1, 1)).cuda()
            self.std=torch.tensor(cfg.DATA.STD).view((1, 3, 1, 1)).cuda()
        """self.mean = torch.tensor([0.457, 0.459, 0.454,0.183]).view((1, 4, 1, 1)).cuda()
        self.std = torch.tensor([0.148, 0.148, 0.154,0.201]).view((1, 4, 1, 1)).cuda()"""

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = int(x1 + crop_sz)

    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = int(y1 + crop_sz)

    x1_pad = int(max(0, -x1))
    x2_pad = int(max(x2 - im.shape[1] + 1, 0))

    y1_pad = int(max(0, -y1))
    y2_pad = int(max(y2 - im.shape[0] + 1, 0))

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)


    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self):
        self.visdom = None

    def predicts_segmentation_mask(self):
        return False

    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')
class MixFormerOnline(BaseTracker):
    def __init__(self):
        self.attn_weights = []
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0


        self.online_size = 1
        self.update_interval = 200
        self.max_score_decay = 1.0

        #print("Search factor: ", self.search_factor)
        print("Update interval is: ", self.update_interval)
        print("Online size is: ", self.online_size)
        print("Max score decay: ", self.max_score_decay)


    def initialize(self, image, info: dict,rt):
        # forward the template once
        self.rt=rt
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], 2.0,
                                                    output_sz=TSIZE)

        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template


        self.online_state = info['init_bbox']
        self.name=info['name']
        if not os.path.exists("rt_imgs_"+self.name):
            os.mkdir("rt_imgs_"+self.name)
        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, 5.0,
                                                                output_sz=SSIZE)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        out_dict = {}
        input1=np.ascontiguousarray(self.template.cpu().numpy().astype(np.float32),dtype=np.float32)
        input2=np.ascontiguousarray(self.online_template.cpu().numpy().astype(np.float32),dtype=np.float32)
        input3=np.ascontiguousarray(search.cpu().numpy().astype(np.float32),dtype=np.float32)
        if self.frame_id==1:
            
            print(input1,input2,input3)
            print(image,self.state)

        with torch.no_grad():
            # TODO: use forward_test() in test
            
            
            input=[input1,input2,input3]
            output = self.rt.track(input)

        pred_boxes = torch.from_numpy(output[0]).reshape(-1, 4)
        pred_score = torch.from_numpy(output[1]).reshape(-1).sigmoid().item()
        with open("rt_log.txt","a") as file:
            file.write(str(pred_boxes))
            file.write(str(pred_score))
            file.write('\n')
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * SSIZE / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        '''if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        2.0,
                                                        128)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:
            # print(f"frame{self.frame_id}: update")
            if self.online_size == 1:
                self.online_template = self.online_max_template

            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            # print("set online")


            self.max_pred_score = -1
            self.online_max_template = self.template'''

        
            
        
        return {"target_bbox": self.state, "conf_score": pred_score,'img':image}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * SSIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * SSIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormerOnline
def compute_iou(rec0_1,rec0_2):
    rec_1=[0,0,0,0]
    rec_2=[0,0,0,0]
    rec_1[0]=rec0_1[0]
    rec_1[1]=rec0_1[1]
    rec_2[0]=rec0_2[0]
    rec_2[1]=rec0_2[1]
    rec_1[2]=rec0_1[0]+rec0_1[2]
    rec_1[3]=rec0_1[1]+rec0_1[3]
    rec_2[2]=rec0_2[0]+rec0_2[2]
    rec_2[3]=rec0_2[1]+rec0_2[3]
    s_rec1=(rec_1[2]-rec_1[0])*(rec_1[3]-rec_1[1])   #第一个bbox面积 = 长×宽
    s_rec2=(rec_2[2]-rec_2[0])*(rec_2[3]-rec_2[1])   #第二个bbox面积 = 长×宽
    sum_s=s_rec1+s_rec2                              #总面积
    left=max(rec_1[0],rec_2[0])                      #交集左上角顶点横坐标
    right=min(rec_1[2],rec_2[2])                     #交集右下角顶点横坐标
    bottom=max(rec_1[1],rec_2[1])                    #交集左上角顶点纵坐标
    top=min(rec_1[3],rec_2[3])                       #交集右下角顶点纵坐标
    if left >= right or top <= bottom:               #不存在交集的情况
        return 0
    else:
        inter=(right-left)*(top-bottom)              #求交集面积
        iou=(inter/(sum_s-inter))*1.0                #计算IOU
        return iou

def process_result(bboxs,scores,image,id):
    image_BGR = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    a=bboxs[0]
    print(init_a)
    print(a)
    a_related=a/init_a
    colors=[(0,255,0),(0,0,255)]
    for i in range(1):
        if scores[0] > 0.99:
            cv.rectangle(image_BGR, (int(bboxs[i][0]),int(bboxs[i][1])), (int(bboxs[i][0]+bboxs[i][2]),int(bboxs[i][1]+bboxs[i][3])), color=colors[i], thickness=2)
        else:
            cv.rectangle(image_BGR, (0,0), (768,432), color=(255,0,0), thickness=2) 
    iou=compute_iou(bboxs[0],bboxs[1])
    iou_related=iou/init_iou
    if iou_related<=0:
        iou_related=None
    elif scores[0]<min_score or scores[1]<min_score:
        iou_related=None
    lost_flag=1 if scores[0]<min_score else 0
    #cv.rectangle(image_BGR, (int(768/2),400), (int(768/2+768/2*math.log(iou_related)),415), color=(255,0,0), thickness=-1)
    #cv.rectangle(image_BGR, (int(768/2),415), (int(768/2+768/2*math.log(a_related)),430), color=(255,0,0), thickness=-1)
    save_path = os.path.join("rt_imgs", "%04d.jpg" % id)
    cv.imwrite(save_path, image_BGR)
    return a_related,iou_related,lost_flag
def crop_array(arr):
    new_arr = np.copy(arr)  
    new_arr[1] = arr[1] + arr[3] / 2
    new_arr[3] = arr[3] / 2
    return new_arr
if __name__=="__main__":
    seq_dir='D:\\dog\\dog\\tensorrt\\img'
    init_depth=3
    init_bbox=np.array([373, 41, 78, 279])
    init_bbox_2=crop_array(init_bbox)
    info={'init_bbox':init_bbox,'engine_file':"D:\\dog\\dog\\tensorrt\\base_body.engine",'name':'body'}
    info_2={'init_bbox':init_bbox_2,'engine_file':"D:\\dog\\dog\\tensorrt\\base_leg.engine",'name':'leg'}
    file_names=os.listdir(seq_dir)
    file_names.sort()
    file_paths=[os.path.join(seq_dir,file_name) for file_name in file_names]
    first_frame=_read_image(file_paths[0])
    tracker = MixFormerOnline()
    tracker_2 = MixFormerOnline()
    rt_2=TensorRTTracker("D:\\dog\\dog\\tensorrt\\base_leg.engine")
    rt=TensorRTTracker("D:\\dog\\dog\\tensorrt\\base_body.engine")
    tracker.initialize(first_frame,info,rt)
    tracker_2.initialize(first_frame,info_2,rt_2)
    rt_2.print_trt_file_name()
    for i in range(len(file_paths)-1):
        bboxs=[]
        scores=[]
        result=tracker.track(_read_image(file_paths[i+1]),info)
        bboxs.append(result['target_bbox'])
        scores.append(result["conf_score"])
        #print(result)
        result_2=tracker_2.track(_read_image(file_paths[i+1]),info_2)
        bboxs.append(result_2['target_bbox'])
        scores.append(result_2["conf_score"])
        a_related,iou_related,lost_flag=process_result(bboxs,scores,result['img'],i)
        if i==0:
            print(bboxs)
            bbox1,bbox2=bboxs[0],bboxs[1]
            init_iou=compute_iou(bbox1,bbox2)
            print(bboxs)
            init_a=bboxs[0][3]
        #print(str(i)+" "+str(result_2))
        
    exit()
    
    