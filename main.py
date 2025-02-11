import os
import argparse
import numpy as np
import tqdm
import clip
import gzip
import pandas as pd
from datetime import datetime
from mapie.conformity_scores.sets.lac import LACConformityScore

import datasets.hmdb51 as hmdb51
import datasets.ucf101 as ucf101
import datasets.kinetics400 as kinetics400
from datasets.utils import calib_test_sets_folds, get_soft_labels
from setup import *

from conformal.lac_fast_methods import fast_fit, fast_get_set

def save_csv_with_progress(df:pd.DataFrame, filename:str, chunksize:int=100000) -> None:
    """
    Save a large DataFrame to a CSV file with a progress bar.

    Parameters:
    - df: DataFrame to be saved
    - filename: Output CSV file path
    - chunksize: Number of rows per chunk (default is 100000)
    """
    n_chunks = len(df) // chunksize + (1 if len(df) % chunksize != 0 else 0)
    
    with gzip.open(filename, "wt", newline='') as f:
        df.iloc[:0].to_csv(f)
        
        for i in tqdm.tqdm(range(n_chunks), desc="Saving DataFrame", unit="chunk"):
            chunk = df.iloc[i * chunksize: (i + 1) * chunksize]
            chunk.to_csv(f, header=False)
    
    print(f"CSV saved successfully as {filename}")

def parse_range(range_str):
    """Parses a string of the format 'min:max:step' into a list of values."""
    try:
        min_val, max_val, step = map(float, range_str.split(':'))
        scale = 10 ** (-np.floor(np.log10(step)).astype(int))
        return np.arange(round(min_val * scale), round(max_val * scale) + 1, round(step * scale)) / scale
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in the format 'min:max:step' with numeric values.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replicating the paper experiments.")
    parser.add_argument("--dataset", default="ucf101", type=str, help="Dataset used. Must be located in 'data' folder and have a corresponding AbstractDataloader object to handle it.")
    parser.add_argument("--backbone", default="ViT-B/16", type=str, help="CLIP backbone used.")
    parser.add_argument("--n_folds", default=40, type=int, help="Number of folds to repeat the experiment and obtain less noisy results.")
    parser.add_argument("--n_shots", default=10, type=int, help="Number of shots for each label in the calibration set.")
    parser.add_argument("--temperatures_grid", default='10:200:5', type=parse_range, help="Range of temperatures tested. Define the range as 'min(incl):max(incl):step', e.g., '10:201:5'.")
    parser.add_argument("--alphas_grid", default='0.01:0.15:0.02', type=parse_range, help="Range of alphas tested. Define the range as 'min(incl):max(incl):step', e.g., '0.01:0.15:0.02'.")
    args = parser.parse_args()

    dataloader_dict = {"ucf101":ucf101.UCF101, "hmdb51":hmdb51.HMDB51, "kinetics400":kinetics400.Kinetics400}
    if args.dataset not in dataloader_dict.keys():
        raise ValueError(
            f"""The submitted dataset '{args.dataset}' is not implemented. 
            Please implement a child class of AbstractDataloader to handle 
            your dataset and add it to 'dataloader_dict'"""
        )
    
    dataset = dataloader_dict[args.dataset](os.path.join(PATH_TO_DATASETS, args.dataset))
    all_features, all_labels = dataset.load_features_and_labels(args.backbone)[:2]

    train_features_folds, train_labels_folds, \
    test_features_folds, test_labels_folds = calib_test_sets_folds(all_features, all_labels, n_shots=args.n_shots, n_folds=args.n_folds)
    
    text_embeddings=dataset.get_textual_prototypes(clip.load(args.backbone)[0])

    logs_list = []
    for fold, (train_features, train_labels, test_features, test_labels) in \
        enumerate(zip(train_features_folds, train_labels_folds, test_features_folds, test_labels_folds)):

        pbar = tqdm.tqdm(total=len(args.temperatures_grid.astype(int)), desc=f"Fold n°{fold+1}/{args.n_folds}")
        for t, temperature in enumerate(args.temperatures_grid):
            train_soft_labels = get_soft_labels(temperature, train_features, text_embeddings)
            test_soft_labels = get_soft_labels(temperature, test_features, text_embeddings)
            
            fitted_scores = fast_fit(LACConformityScore(), train_soft_labels, train_labels, args.alphas_grid)

            for a, alpha in enumerate(args.alphas_grid):        
                y_set = fast_get_set(fitted_scores, test_soft_labels, alpha)
                
                quantile = fitted_scores.quantiles_
                set_sizes = y_set.sum(axis=1)

                temp_logs = pd.DataFrame({
                    "fold":fold, 
                    "temperature":temperature, 
                    "alpha":alpha, 
                    "quantile":quantile, 
                    "set_sizes":set_sizes
                })
                temp_logs.index.name="sample"
                logs_list.append(temp_logs)

            pbar.update(1)
        pbar.close()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Fusing experiments logs...", end="", flush=True)
    logs = pd.concat(logs_list)
    print(" Done", flush=True)
    save_csv_with_progress(logs, os.path.join(PATH_TO_RESULTS, f"results_{args.dataset}_{timestamp}.csv.gz"), chunksize=50_000)
