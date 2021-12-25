# Getting Started
This page provides basic tutorials about the usage of boxlevelset.
For installation instructions, please see [INSTALL.md](INSTALL.md).


## Prepare iSAID dataset.
It is recommended to symlink the dataset root to `boxlevelset/data`.

The detailed introduction of iSAID dataset preparation can refer to this  official [repo](https://github.com/CAPTAIN-WHU/iSAID_Devkit).

**Note:** We split iSAID images into all the `800x800` patches for each `original image`, `color_RGB image` and `id_color image`.  The `.py` script can use the `DOTA_devkit/SplitOnlyImage_multi_process.py`, instead of the official [split.py](https://github.com/CAPTAIN-WHU/iSAID_Devkit/blob/master/preprocess/split.py)  that is less than or equal to 800 with different patches size.

The structure of patches data:
```shell
iSAID_pathches_800
    ├── train
    │   └── images
    │       ├── P0000__1__0___0_instance_color_RGB.png
    │       ├── P0000__1__0___0_instance_id_RGB.png
    │       ├── P0000__1__0___0.png
    │       ├── ...
    │       ├── P0010__1__0___1200_instance_color_RGB.png
    │       ├── P0010__1__0___1200_instance_id_RGB.png
    │       └── P0010__1__0___1200.png
    └── val
    |    └── images
    |        ├── P0003__1__0___0_instance_color_RGB.png
    |        ├── P0003__1__0___0_instance_id_RGB.png
    |        ├── P0003__1__0___0.png
    |        ├── ...
    ├── test
    │   └── images
    │       ├── P0006__1__0__0.png
    │       └── ...
```

Then, we need to generate coco-format **.json** annotation files for train and val split images by running `python preprocess.py --set train,val` in this  official [repo](https://github.com/CAPTAIN-WHU/iSAID_Devkit).  (So you need to install the right envs for this [repo](https://github.com/CAPTAIN-WHU/iSAID_Devkit).)


 
## Train a model
mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**Important**: The default learning rate in config files is for 4 GPUs.
If you use less or more than 4 GPUs, you need to set the learning rate proportional
to the GPU num, e.g., 0.01 for 8 GPUs and 0.02 for 16 GPUs.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
example:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/BoxSegmentation/boxlevelset_re50_refpn_1x_isaid_800_800.py 4
```

Optional arguments are:

- `--validate` (recommended): Perform evaluation at every k (default=1) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

# Test 

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]

# test boxlevelset 
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]
```

1. Test boxlevelset.
```shell
python tools/test.py configs/BoxSegmentation/boxlevelset_re50_refpn_1x_isaid_800_800.py \
    work_dirs/boxlevelset_re50_refpn_isaid/epoch_12.pth \ 
    --out work_dirs/boxlevelset_re50_refpn_isaid/results.pkl --eval bbox segm
```

2. Test boxlevelset with 4 GPUs.
```shell
./tools/dist_test.sh configs/BoxSegmentation/boxlevelset_re50_refpn_1x_isaid_800_800.py \
    work_dirs/boxlevelset_re50_refpn_isaid/epoch_12.pth \ 
    --out work_dirs/boxlevelset_re50_refpn_isaid/results.pkl --eval bbox segm
```


### Demo for test visualization .
    
```shell
cd demo
python demo_inference.py  ../configs/BoxSegmentation/boxlevelset_re50_refpn_1x_isaid_800_800.py ../work_dirs/boxlevelset_re50_refpn_isaid/epoch_12.pth  test_img_path  show_mask_img_path
```
