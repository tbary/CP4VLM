# -*- coding: utf-8 -*-
import argparse
import os
import datasets.hmdb51 as hmdb51
import datasets.ucf101 as ucf101
import datasets.kinetics400 as kinetics400
import clip

dataloader_dict = {"ucf101":ucf101.UCF101, "hmdb51":hmdb51.HMDB51, "kinetics400":kinetics400.Kinetics400}
backbones = {'vit_b16':'ViT-B/16',
             'vit_l14':'ViT-L/14',
             'vit_b32':'ViT-B/32',
             'rn50':'RN50',
             'rn101':'RN101'}
def main(args):
    dataset = dataloader_dict[args.dataset](os.path.join(args.data_root_path, args.dataset))
    print('Starting frames extraction....')
    dataset.ExtractFrames()
    print('Frames extraction done.\n')
    
    clip_model, preprocess = clip.load(backbones[args.backbone_name])
    
    print('Starting frames encoding....')
    dataset.EncodeFrames(clip_model, args.backbone_name, preprocess,)
    print('Frames encoding done.\n')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preparing datsets : frame extraction and encoding.")
    parser.add_argument('--data_root_path', type = 'str', help="Root path where to find the dataset.")
    parser.add_argument('--frames_output_path', type = 'str', default = None, 
                        help = 'Path were extracted frames are stored. Defaults to data_root_path/{dataset}/frames/ internally.')
    parser.add_argument('--features_output_path', type = 'str', default = None, 
                        help = 'Path were visual features of frames are stored. Defaults to data_root_path/{dataset}/cache/ internally.')
    parser.add_argument("--dataset", default="ucf101", type=str, help="Dataset used. Must be located in 'data' folder and have an AbstractDataloader object to handle it.")
    parser.add_argument('--backbone_name', default = 'vit_b16', type = str,
                        help = "Backbone to use. One of 'vit_b16', 'vit_l14', 'vit_b32', 'rn50' or 'rn101'.")
    
    parser.add_argument("--num_frames_per_video", default = 10, type = int, 
                        help = "Number of frames to extract and encode for each video clip.")
    args = parser.parse_args()
    
    
    
    