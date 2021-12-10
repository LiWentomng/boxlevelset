## Installation

### Requirements

- Linux
- Python 3.5/3.6/3.7
- PyTorch 1.1/1.3.1
- CUDA 10.0/10.1
- NCCL 2+
- GCC 4.9+
- [mmcv<=0.2.14](https://github.com/open-mmlab/mmcv)

We have test the following environment:
```shell
Python 3.7
Pytorch 1.1.0
CUDA 10.0
mmcv==0.2.13
```


### Install boxlevelset

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n boxlevelset python=3.7 -y
source activate boxlevelset

conda install cython
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```
conda install pytorch=1.3.1 torchvision cudatoolkit=10.0 -c pytorch -y
```

Note:
1. If you want to use Pytorch>1.5, you have to made some modifications to the `cuda ops`. See [here](https://github.com/csuhan/ReDet/issues/1) for a reference.
2. There is a known [bug](https://github.com/csuhan/ReDet/issues/4) happened to some users but not all (As I have successfully run it on V100 and Titan Xp). If it occurs, please refer to [here](https://github.com/csuhan/ReDet/issues/4).

c. Clone the boxlevelset repository.

```shell
https://github.com/LiWentomng/boxlevelset.git
cd boxlevelset
```

d. Compile cuda extensions.

```shell
bash compile.sh
```

e. Install boxlevelset (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -e ."
```

### Install DOTA_devkit
```
sudo apt-get install swig
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
