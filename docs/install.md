## Environments

- Linux or macOS
- Python 3.6+  (Python 3.7 in our envs)
- PyTorch 1.3+ (1.7.1 in our envs)
- CUDA 9.2+ (CUDA 10.1 in our envs)
- GCC 5+
- mmdet==2.10.0
- mmcv-full==1.3.13 ([MMCV](https://mmcv.readthedocs.io/en/latest/#installation))

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
    
    ```shell
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
    conda install -c conda-forge cudatoolkit-dev=10.1       #this is required and needs long time during the installation. 
   ```
    Note: Make sure that your compilation CUDA version and runtime CUDA version match. We use the PyTorch 1.7.1
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).


3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    
    #Example in our envs.
    pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
   ```
    Note: See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Here we use the mmcv-full=1.3.13

4. Clone the MMDetection repository.

    ```shell
    git clone https://github.com/LiWentomng/boxlevelset.git
    cd boxlevelset
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    bash setup.sh   #compile the whole envs 
    ```

Note:

a. Following the above instructions, MMDetection is installed on `dev` mode
, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv
-python`,
you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `bash setup.sh` will
 only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


