## Performance of  models  (ECCV2022 paper)
### Models 
 * The following models are trained with Telsa V100 GPU. 
 * The pretrained models are in GoogleDriver.

#### Mask AP Results on Pascal VOC val
|     Backbone    |  schd | Models | GPUs | AP  | AP_25 | AP_50 | AP_70 | AP_75 | 
|:---------------:|--------|--------|:----:|:----:|:-----:|:-----:|:--------:|:---------:|
|    ResNet-50    |   3x   |[model](https://drive.google.com/file/d/1Yl4QCRx_VKY_OvEI6sz36BI88sSOWWhu/view?usp=sharing) |  4 |36.5 |  76.8 |  64.2 |   44.8   |    36.4   |  
|   ResNet-101    |   3x   |[model](https://drive.google.com/file/d/1gMWGxmPyHFyxR0re3lHbMjxl3xvPvWeh/view?usp=sharing) |  4 |38.3 |  77.9 |  66.3 |   46.4   |    38.7   | 


#### Mask AP Results on COCO 2017
|     Backbone    |  schd  |Models | GPUs | AP(val)  | AP(test-dev) |
|:---------------:|--------|--------|:----:|:----:|:-----:|
|    ResNet-50    |  3x    |[model](https://drive.google.com/file/d/1R-2s5wh-Rj82yieFcXa5_T9gW9oB29dj/view?usp=sharing)  |  8  |31.4 |  31.7 | 
|   ResNet-101    |  3x    |[model](https://drive.google.com/file/d/1mZ5PBRINlfhHPxzSPvace65Qs4lzG2kL/view?usp=sharing)  |  8  |33.0 |  33.4 | 
|   ResNet-101-DCN|  3x    |[model](https://drive.google.com/file/d/1aZN9CUd2flcsW_KewWUgerjF0AjWszCB/view?usp=sharing)  |  8  |35.0 |  35.4 | 
 
 Note: 
 * Following [BBTP](https://github.com/chengchunhsu/WSIS_BBTP) and [DiscoBox](https://github.com/NVlabs/DiscoBox), the Pascal VOC is aumented Pascal VOC([data link](https://drive.google.com/file/d/16Mz13NSZBbhwPuRxiwi7ZA2Qvt9DaKtN/view?usp=sharing)) with SBD. We recomment the users to train the Pascal VOC first to validate the performance with  ~14 hours training time. 
 * Training COCO with 3x needs about 4 days. 
