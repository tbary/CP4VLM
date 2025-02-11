# -*- coding: utf-8 -*-
import torch
import os
import pickle
import clip
import json
import cv2
from tqdm import tqdm
from PIL import Image, ImageOps
import pandas as pd
from .utils import AbstractDataloader
current_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # dirty but compatible with python < 3.9
#%%
class UCF101(AbstractDataloader):
    
    def get_labels_names(self):
        with open(os.path.join(self.root_path, 'split_zhou_UCF101.json'), 'r') as f:
            zhou_split = json.load(f)
        li_labels = []
        for u in zhou_split['test']:
            if u[-1] not in li_labels:
                li_labels.append(u[-1])
        li_labels = sorted(li_labels)
        self.labels_names = li_labels
        self.labels_to_idx = {}
        self.idx_to_labels = {}
        for jlabel, label in enumerate(li_labels):
            self.labels_to_idx[label] = jlabel
            self.idx_to_labels[jlabel] = label
        return None
    
    def get_label_dict(self, splits = ['test', 'val']):
        li_dicts = []
        for jsplit, split in enumerate(splits):
            path = os.path.join(self.root_path, 'annotations',f'{split}.csv')
            df = pd.read_csv(path)
            label_names_dict = {}
            for index, row in tqdm(df.iterrows()):
                if row['youtube_id']:
                    label_names_dict[row['youtube_id']] = row['label']
                    
            li_dicts.append(label_names_dict)
            setattr(self, f'label_dict_{split}', label_names_dict)
            
        return li_dicts
    
    
    def ExtractFrames(self, num_frames = 10, output_folder = None):
        if output_folder is None:
            output_folder = os.path.join(self.root_path, 'frames')
        os.makedirs(output_folder, exist_ok=True)    
            
        if len(os.listdir(output_folder))>0:
            print(f'Output folder {output_folder} is non-empty. Assuming frames extraction has already been performed.')
            return None
        
        for jlabel, label_ in enumerate(tqdm(self.labels_names)):
            label = label_.replace('_', '')
            base_directory = os.path.join(self.root_path, label)
            label_save_path = os.path.join(self.root_path, 'frames', label)
            os.makedirs(label_save_path, exist_ok=True)
            li_extracted = {u:'' for u in os.listdir(label_save_path)}
            
            for file in os.listdir(base_directory):
                if file.endswith(".avi"):
                    filename = file.split('.avi')[0]
                    if filename not in li_extracted:
                        video_path = os.path.join(base_directory, file)
                        # Create a folder for the frames of this video
                        video_output_folder = os.path.join(output_folder, label, filename)
                        os.makedirs(video_output_folder, exist_ok=True)
                        self.__extract_frames(video_output_folder = video_output_folder, 
                                              video_path = video_path,
                                              num_frames = num_frames)
                    
        return None
    
    def __get_label_path_from_label_name(self, label_name):
        return label_name.replace('_', '')
    
    def EncodeFrames(self, 
                     clip_model, 
                     backbone_name,
                     preprocess,
                     cache_dir = None,
                     img_size = (224,224), 
                     samples_batch_size = 20, 
                     frames_per_clip = 10, 
                     model_dim = 512,
                     force_overwrite = False):
        bname = backbone_name.replace('/', '_').replace('-','_')
        if cache_dir is None:
            cache_dir = os.path.join(self.root_path, 'cache')
        path = os.path.join(cache_dir, f'{bname}_features.pickle')
        if os.path.exists(path):
            if not force_overwrite:
                raise RuntimeError(f'Found cached feature for backone {backbone_name}. Use force_overwrite to overwrite them.')
            else:
                print(f'Found cached feature for backone {backbone_name}. They will be overwritten.')

        features = {}
        features_filenames = []
        t_ims = []
        jsample = 0
        batch_num = 0
        for jlabel, label_ in enumerate(tqdm(self.labels_names)):
            label = self.__get_label_path_from_label_name(label_)
            label_path = os.path.join(self.root_path, 'frames', label)
            
            for jdi, di in enumerate(os.listdir(label_path)):
                path = os.path.join(label_path, di)    
                try:  
                    li_frames = list(os.listdir(path))
                except FileNotFoundError:
                    print(f'Frames not found for file {di}. Skipping it.')
                    li_frames = []
                li_idx_frames = torch.tensor([int(u.split('.jpeg')[0].split('frame_')[-1]) for u in li_frames])     
                if len(li_idx_frames) >= frames_per_clip:                
                    idx_sort_frames = torch.argsort(li_idx_frames)
                    for jsort in idx_sort_frames:
                        frame = li_frames[jsort]
                        im = Image.open(os.path.join(path, frame))
                        p_im = preprocess(im)
                        t_ims.append(p_im)
                    features_filenames.append(di)
                
                
                if (len(t_ims) == samples_batch_size) or (jdi == len(os.listdir(label_path))-1):
                    if jsample > 0:
                        t_ims = torch.stack(t_ims)
                        with torch.no_grad(), torch.autocast('cuda', dtype = torch.float16):
                            encoded = clip_model.visual(t_ims.cuda())
                        encoded = encoded.reshape((encoded.shape[0]//frames_per_clip,frames_per_clip,model_dim))
                        batch_num += 1
                        t_ims = []
                        for jfilename, filename in enumerate(features_filenames):
                            features[filename] = encoded[jfilename,...].cpu()
                        features_filenames = []
                
                jsample += 1
                
        os.makedirs(cache_dir, exist_ok=True)           
        with open(os.path.join(cache_dir, f'{bname}_features.pickle'), 'wb') as f:
            pickle.dump(features, f)  
        return None
    
    def load_features_and_labels(self, backbone_name, split_data = False,  splits = ['test', 'val', 'train']):
        bname = backbone_name.replace('/', '_').replace('-','_')
       
        path = os.path.join(self.root_path , 'cache',f'{bname}_features.pickle')
        out = []
        with open(path, 'rb') as f:
            features = pickle.load(f)
            
        with open(os.path.join(self.root_path, 'split_zhou_UCF101.json'), 'r') as f:
            zhou_split = json.load(f)
        self.get_labels_names()
        out = []
        if not split_data:
            all_features = []
            all_labels = []
        for jsplit, split in enumerate(splits):
            if split_data:
                split_features = []
                split_labels = []
            for jfile, filename_ in enumerate(zhou_split[split]):
                filename = filename_[0].split('/')[-1].split('.jpg')[0]
                if split_data:
                    try:
                        split_features.append(features[filename])
                        split_labels.append(self.labels_to_idx[filename_[-1]])
                    except KeyError:
                        print(f'File {filename} has not been encoded.')
                else:
                    try:
                        all_features.append(features[filename])
                        all_labels.append(self.labels_to_idx[filename_[-1]])
                    except KeyError:
                        print(f'File {filename} has not been encoded.')
            if split_data:
                split_features = torch.stack(split_features)
                out.append(split_features)
                out.append(torch.tensor(split_labels))
        if not split_data:
            all_features = torch.stack(all_features)
            out = [all_features, torch.tensor(all_labels)] + [None for _ in range(2*(len(splits)-1))]
        return out
