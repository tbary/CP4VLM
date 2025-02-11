
# -*- coding: utf-8 -*-
import torch
import os
import pickle
from tqdm import tqdm
from PIL import Image
import pandas as pd
from .utils import AbstractDataloader
import cv2
#%%

class Kinetics400(AbstractDataloader):   
        
    def get_labels_names(self):
        split = 'val'
        if not hasattr(self, f'label_dict_{split}'):
            self.__get_label_dict(splits = [split])
        label_names = []
        for f in self.label_dict_val.keys():
            l = self.label_dict_val[f]
            if l not in label_names:
                label_names.append(l)
        label_names = sorted(label_names) #this is dumb but im lazy ; good way should be keeping label_names sorted at all time
        self.labels_names = label_names
        self.labels_to_idx = {}
        self.idx_to_labels = {}
        for jl,l in enumerate(label_names):
            self.labels_to_idx[l] = jl
            self.idx_to_labels[jl] = l
        return None
    
    def __get_label_dict(self, splits = ['test', 'val']):
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
    
    def ExtractFrames(self, split = 'test', num_frames = 10, output_folder = None, continue_extraction = False):
        if output_folder is None:
            output_folder = os.path.join(self.root_path, 'frames')
        os.makedirs(output_folder, exist_ok=True)    
            
        if len(os.listdir(output_folder))>0:
            print(f'Output folder {output_folder} is non-empty.')
            if not continue_extraction:
                print('Since continue_extraction = {continue_extraction}, we assume frame extraction has already been performed.')
                return None
            else:
                print('Since continue_extraction = {continue_extraction}, we continue frame extraction on remaining videos..')
        
        li_extracted = {u:'' for u in os.listdir(output_folder)}
        base_directory = os.path.join(self.root_path, split)
        ignored_files = ['i4qFc-2RU18_000055_000065', 'I0luMKjIZyg_000422_000432', 'QfuO07EqYhI_000054_000064',
                         '084k_RL3ApU_000109_000119','74iWTzKsHPI_000110_000120','jJFqy6yiXzQ_000024_000034',
                         'y7cYaYX4gdw_000047_000057', 'z35QkFl2tyU_000026_000036'] # corrupted files w/o replacement
        for root, _, files in os.walk(base_directory):
            for file in tqdm(files):
                if file.endswith(".mp4") and file.split('.mp4')[0] not in ignored_files:
                    yt_id = '_'.join(file.split('_')[:-2])
                    try:
                        li_extracted[yt_id]
                    except KeyError:
                        video_path = os.path.join(root, file)
                        # Create a folder for the frames of this video
                        video_output_folder = os.path.join(output_folder, yt_id)
                        os.makedirs(video_output_folder, exist_ok=True)
                        print(f"Processing video: {video_path}")
                        self.__extract_frames(video_output_folder = video_output_folder, 
                                              video_path = video_path,
                                              num_frames = num_frames)
        return None
    
    
    def EncodeFrames(self, 
                     clip_model, 
                     backbone_name,
                     preprocess,
                     cache_dir = None,
                     img_size = (224,224), 
                     samples_batch_size = 20, 
                     frames_per_clip = 10, 
                     split = 'test',
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
        split_path = os.path.join(self.root_path, split + '_frames')
        
        for jdi, di in enumerate(tqdm(os.listdir(split_path))):
            path = os.path.join(split_path, di)    
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
            
            if (len(t_ims) == samples_batch_size) or (jdi == len(os.listdir(split_path))-1):
                t_ims = torch.stack(t_ims)
                with torch.no_grad(), torch.autocast('cuda', dtype = torch.float16):
                    encoded = clip_model.visual(t_ims.cuda())
                encoded = encoded.reshape((encoded.shape[0]//frames_per_clip,frames_per_clip,model_dim))
                t_ims = []
                for jfilename, filename in enumerate(features_filenames):
                    features[filename] = encoded[jfilename,...].cpu()
                features_filenames = []
        os.makedirs(cache_dir, exist_ok=True)           
        with open(os.path.join(cache_dir, f'{bname}_{split}_features.pickle'), 'wb') as f:
            pickle.dump(features, f)  
            
        return None
      
    def load_features_and_labels(self, backbone_name, split_data = False,  splits = ['val','test']):
        out = []
        bname = backbone_name.replace('/', '_').replace('-','_')
        if not(hasattr(self, 'labels_to_idx')):
            self.get_labels_names()
        if not split_data:
            all_features = []
            all_labels = []
        for split in splits:
            split_features = []
            split_labels = []
            path = os.path.join(self.root_path , 'cache',f'{bname}_{split}_features.pickle')
            if not hasattr(self, f'label_dict_{split}'):
                self.__get_label_dict(splits = [split])
            with open(path, 'rb') as f:
                features_ = pickle.load(f)
            for key in features_.keys():
                label_name = getattr(self, f'label_dict_{split}')[key]
                if split_data:
                    split_features.append(features_[key])
                    split_labels.append(self.labels_to_idx[label_name])
                else:
                    all_features.append(features_[key])
                    all_labels.append(self.labels_to_idx[label_name])
            if split_data:
                out.append(torch.stack(split_features))
                out.append(torch.tensor(split_labels))
        if not split_data:
            out = [torch.stack(all_features), torch.tensor(all_labels)] + [None for _ in range(2*(len(splits)-1))]
            
        return out
    
        