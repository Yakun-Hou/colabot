import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import opencv_loader_dev
from lib.train.admin import env_settings


class DEV(BaseVideoDataset):
    """ DEV dataset.

    DOG-EYE VIEW DATASET
    """

    def __init__(self, root=None, image_loader=opencv_loader_dev, split="train",  data_fraction=None,type=0):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.split=split
        self.root=root
        self.image_loader=image_loader
        self.datatype=type
        root = env_settings().dev_dir if root is None else root
        super().__init__('DEV', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

    def get_name(self):
        return 'DEV'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)



    def _get_sequence_list(self):
        seqs=os.listdir(os.path.join(self.root, '{}/'.format(self.split)))
        if self.datatype==0:
            seqs=list(filter(lambda x:x.startswith('0'),seqs))
        else:
            seqs=list(filter(lambda x:x.startswith('1'),seqs))
        print(seqs)

        return seqs
    def _get_frame_name_list(self,seq_id):
        frame_names=os.listdir(os.path.join(self.root, '{}/'.format(self.split),self.sequence_list[seq_id], "annotation/"))
        frame_names=[onename.split(".")[0] for onename in frame_names]
        frame_names.sort()
        return frame_names

    def _read_bb_anno(self, seq_id,anno_type="bbox",frame_id=0):
        if anno_type=="depth":
            anno_type=1
            anno_files=[self._get_frame_name_list(seq_id)[frame_id]]
        else:
            anno_type=0
            anno_files=self._get_frame_name_list(seq_id)
        bb_anno_file = os.path.join(self.root, '{}/'.format(self.split),self.sequence_list[seq_id], "annotation/")
        
        anno_infos=[]
        for onefile in anno_files:
            with open(os.path.join(bb_anno_file, onefile+".txt")) as f:
                # extract bbox
                single_info = f.readlines()
                if single_info==[]:
                    print(os.path.join(bb_anno_file, onefile+".txt"))
                #print(single_info)
                single_info=single_info[anno_type].split(' ')[0:4]
                
                minx=min(int(single_info[0]),int(single_info[2]))
                miny=min(int(single_info[1]),int(single_info[3]))
                maxx=max(int(single_info[0]),int(single_info[2]))
                maxy=max(int(single_info[1]),int(single_info[3]))
                w=maxx-minx
                h=maxy-miny
                single_info[0]=minx if minx>0 else 0
                single_info[1]=miny if miny>0 else 0
                single_info[2]=w
                single_info[3]=h
                anno_infos.append(single_info)
        
        return torch.tensor(anno_infos)
    def _read_visible_anno(self, seq_id,frame_id=None):
        #print(1)
        bb_anno_file = os.path.join(self.root, '{}/'.format(self.split),self.sequence_list[seq_id], "annotation/")
        if frame_id==None:
            anno_files=self._get_frame_name_list(seq_id)
        else:
            anno_files=[self._get_frame_name_list(seq_id)[i] for i in frame_id]
        anno_infos=[]
        #print(anno_files)
        for onefile in anno_files:
            with open(os.path.join(bb_anno_file, onefile+".txt")) as f:
                # extract bbox
                line=f.readlines()[0][0:-1].split(' ')[4]
                single_info = 1 if str(line)=="0" else 0
                #print(f.readlines()[0][0:-1].split(' '))
                anno_infos.append(single_info)
                #print(line)
        
        return torch.tensor(anno_infos)
    def _read_class_id(self, seq_id):
        bb_anno_file = os.path.join(self.root, '{}/'.format(self.split),self.sequence_list[seq_id], "annotation/")
        anno_files=os.listdir(bb_anno_file)
        anno_infos=[]

        with open(os.path.join(bb_anno_file, anno_files[0])) as f:
            return (f.readlines()[0].split(' ')[4])
        return '0'


    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, '{}/'.format(self.split), self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_visible_anno(seq_id).byte()
        #print("before: "+str(visible))
        #visible = valid.clone().byte()
        #print("after : "+str(visible))
        info={'bbox': bbox, 'valid': valid, 'visible': visible}
        return info



    def _get_frame_path(self, seq_path, frame_name):
        
        return [os.path.join(seq_path,'img', frame_name+'.png'),os.path.join(seq_path,'d', frame_name+'.tif')]    # frames start from 0

    def _get_frame(self, seq_path, frame_id):
        
        frame_rgb=torch.tensor(self.image_loader(self._get_frame_path(seq_path, frame_id)[0]))
        frame_d=torch.tensor(self.image_loader(self._get_frame_path(seq_path, frame_id)[1]))*2.0
        #print(frame_d.shape)
        frame_d=frame_d.reshape(frame_d.shape[0],frame_d.shape[1],1)
        #frame=torch.cat([frame_rgb,frame_d],dim=2)
        frame=frame_rgb
        return frame.numpy()

    def _get_depth(self,seq_id,frame_id):
        seq_path = self._get_sequence_path(seq_id)
        #frame_path=(self._get_frame_path(seq_path, self._get_frame_name_list(seq_id)[frame_id]))
        #frame_d=torch.tensor(self.image_loader(frame_path[1]))*2.0
        depth_sample_pos=self._read_bb_anno(seq_id,anno_type="depth",frame_id=frame_id)[0]
        x_c=(int(depth_sample_pos[0])+0.5*int(depth_sample_pos[2]))/768.0
        y_c=(int(depth_sample_pos[1])+0.5*int(depth_sample_pos[3]))/432.0
        #print(frame_d.shape[0],frame_d.shape[1])
        #frame_d=frame_d.reshape(frame_d.shape[0],frame_d.shape[1],1)
        return [x_c,y_c]
    def get_class_name(self, seq_id):
        classid=_read_class_id(seq_id)
        cls_name = {}
        with open(os.path.join(self.root, 'meta.txt')) as f:
            anno_info = f.readlines()
        for single_info in anno_info:
            cls_name[(single_info.split(' ')[0])]=single_info.split(' ')[-1]
        return cls_name[classid]

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_name_list=self._get_frame_name_list(seq_id)
        frame_list = [self._get_frame(seq_path, frame_name_list[f_id]) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        if len(frame_list)==0:
            print(seq_id)
            assert len(frame_list)>0
        return frame_list, anno_frames #, obj_meta