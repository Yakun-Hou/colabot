import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class DEVDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split='val'):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.dev_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.dev_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name,'annotation/')
        ground_truth_rect=[]
        file_list=os.listdir(anno_path)
        file_list.sort()
        for single_file in file_list:
            file_path=os.path.join(anno_path,single_file)
            #print(anno_path)
            with open(file_path) as f:
                txt=f.readlines()[0].replace('\n','').split(' ')
                print(txt)
                single_info_withflag = txt[0:5]
                single_info=single_info_withflag[0:4]
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
                if int(single_info_withflag[-1])!=0:
                    single_info=[0,0,999,999]
                ground_truth_rect.append(single_info)

        frames_path_rgb = os.path.join(self.base_path, sequence_name,'img')
        frames_path_d = os.path.join(self.base_path, sequence_name,'d')
        frame_list_rgb = os.listdir(frames_path_rgb)
        frame_list_d = os.listdir(frames_path_d)
        frame_list_rgb.sort(key=lambda f: int(f.split('.')[0]))
        frame_list_d.sort(key=lambda f: int(f.split('.')[0]))
        #frames_list_rgb = [os.path.join(frames_path_rgb, frame) for frame in frame_list_rgb]
        #frames_list_d = [os.path.join(frames_path_d, frame) for frame in frame_list_d]
        #frames_list=[[os.path.join(frames_path_rgb, frame_rgb),os.path.join(frames_path_d, frame_d)] for frame_rgb,frame_d in zip(frame_list_rgb,frame_list_d)]
        frames_list=[[os.path.join(frames_path_rgb, frame_rgb),os.path.join(frames_path_d, frame_d)] for frame_rgb,frame_d in zip(frame_list_rgb,frame_list_d)]
        return Sequence(sequence_name, frames_list, 'dev', np.array(ground_truth_rect).reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        ldir=os.listdir(self.base_path)
        
        return ldir
