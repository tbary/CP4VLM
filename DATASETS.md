# How to install datasets
All datasets should be under the same folder (say `$DATA`) and organized as follow to avoid modifying the source code. The file structure looks like
```
$DATA/
├── Kinetics400/
├── UCF101/
└── hmdb51/
```
If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [Kinetics400](#Kinetics400)
- [UCF101](#UCF101)
- [hmdb51](#HMDB51)

Once you have installed a dataset, you can prepare it (Frames extraction and Frames encoding) with prepare_datasets.py found in the root of our repository:
```
python prepare_datasets.py --root_data_path $DATA --dataset "dataset_name" --backbone "vit_b16"
```
dataset_name is either kinetics400, hmdb51 or ucf101.
The frames will only be extracted the first time you run the command line for each dataset. You can loop over all available backbones:
```
backbones=("vit_b16" "vit_l14" "vit_b32" "rn50" "rn101")
for backbone in "${backbones[@]}"; do
    echo "Processing with backbone: $backbone"
    python prepare_datasets.py --root_data_path "$DATA" --dataset "dataset_name" --backbone "$backbone"
done
```
### Kinetics400
- Create a folder named `Kinetics400/` under `$DATA`.
- Clone https://github.com/cvdfoundation/kinetics-dataset into it.
- Comment the parts concerning the training set in k400_downloader.sh and k400_extractor.sh
- Run k400_downloader.sh
- Run k400_extractor.sh

You should end up with a file structure that looks like this
```
Kinetics400/
--kinetics-dataset
    -- k400_targz
        --...
    -- k400
        -- annotations
        -- test
        -- val
```

### UCF101
Go to https://www.crcv.ucf.edu/data/UCF101.php and download the rar archive into "$DATA/UCF101". Decompress it and you should end up with a file structure that looks like this:
```
UCF101/
    -- ApplyEyeMakeup
    -- ApplyLipstick
    --...
```


Then, go to https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#ucf101 and download split_zhou_UCF101.json. Place it at '$DATA/UCF101/'.

### HMDB51
Go to https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads and download hmdb51_org.rar into "$DATA/hmdb51" and decompress it.
You should obtain a file structure that looks like this 

```
hmdb51/
    -- brush_hair
    -- cartwheel
    --...
```
Then, download 'three splits for the hmdb51' from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads .
Create a folder 'splits' in $DATA/hmdb51/ and place all the text files inside.