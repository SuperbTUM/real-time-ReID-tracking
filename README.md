## Update

- Feb. 22, Add Video-based training code and switch IoU to DIoU
- Feb. 11, Add IBN and modified MIN_CONF to 0.5
- Feb. 8, 2022 Reformat files
- Feb. 4, 2022  Add some thoughts 
- Jan. 3, 2022  Add more details
- Dec. 20, 2021  First submit



## Introduction

This is a project of real-time multiple object tracking in person re-identification in collaboration with two data science master students. The proposal is to re-design the Re-ID model and try to obtain a stronger backbone. Several thoughts were applied and will be discussed throughly in the following chapters. The baseline is a Yolov5 based DeepSort algorithm and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch). The author has updated his repository with import of fast-reid package. I developed my code with Dec. 2021 version when the re-id model from scratch existed. The baseline backbone was ResNet-18.



## Dependency

The code was developed with PyTorch and Numpy. Please download and install the latest stable version with CUDA 11. All codes are tested with one single GPU.



## Dataset

We use MOT16 as benchmark and Market1501 to train our re-id network.

Videos: [MOT16](https://motchallenge.net/data/MOT16/) => This dataset could be evaluated with completed bash scripts.

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data) => We only use the training set for training. (It is expected to use all the images but we just want to have 751 classes.)



## Quick Start

Please do not use Google Colab, even if you have a Pro account, since file transfer is rather unstable (especially in the evaluation stage)!

Replace the file with modified one in the original repositories.

You can try to have generated images with GAN. That means you need additional training on the GAN. To train DC-GAN, please refer to the instructions on [this](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). Please pay attention that this is an out-of-state repo and there is also file missing. You can also refer to the `ipynb` file in [modification_dcgan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_dcgan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py` before conducting training. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

You will need to train the Re-ID model with Market1501. In the [modification_deepsort](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort) folder, you can see how we build the model as well train the model. The intuition is to use bag of tricks. We considered random erasing augmentation, last stride reduction, center loss, SE block, batch norm neck, etc. We found that IBN module is important, so we also include this in our backbone. We have our checkpoint available [here](https://drive.google.com/file/d/1Ta89D7WXhL_H2lR_eYEyLuWfZEFdNEXo/view?usp=sharing). 

If you want to train the Re-ID model with video dataset, please refer to the [video_train](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort/mot16_train.py) script.

For the final evaluation, please refer to [this Wiki](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation) for details. We also have our script file for tracking (May not reliable as running in Colab).



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The inference speed is 15 ~ 20 ms per frame depending on the sparsity of pedestrians with 640 * 640 resolution with Tesla T4. It may be slower if bounding box is resized to (128, 256). The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 17 ms per frame under the same testing environment. The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. The evaluation can be deployed with a bash command. I set the maximum distance to 0.15 (cosine distance) and minimum confidence to 0.5. Also, I resized the bounding box to (128, 256). The evaluation results are shown below. Our proposal has better performance in MOTA, MOTP, MODA and IDF1 with 1% of absolute improvement! For specific meaning of tracking metrics, please refer to [this](https://link.springer.com/content/pdf/10.1007/s11263-020-01375-2.pdf) and [this](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf).

**Strong Baseline**

|          | MOTA ⬆ | MOTP ⬆ | MODA ⬆ | CLR_Re ⬆ | CLR_Pr ⬆ | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆ | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇ | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇ |
| -------- | ------ | ------ | ------ | -------- | -------- | ------ | ------ | ------ | ------- | -------- | -------- | -------- | ------ | ---- | ---- | ---- | ------ |
| MOT16-02 | 35.188 | 77.36  | 35.748 | 40.279   | 89.889   | 16.667 | 42.593 | 40.741 | 26.068  | 7183     | 10650    | 808      | 100    | 9    | 23   | 22   | 287    |
| MOT16-04 | 61.171 | 76.199 | 61.362 | 73.379   | 85.928   | 42.169 | 42.169 | 15.663 | 43.706  | 34897    | 12660    | 5715     | 91     | 35   | 35   | 13   | 468    |
| MOT16-05 | 57.48  | 78.314 | 58.463 | 70.035   | 85.82    | 31.2   | 55.2   | 13.6   | 42.293  | 4775     | 2043     | 789      | 67     | 39   | 69   | 17   | 181    |
| MOT16-09 | 60.148 | 83.663 | 60.985 | 75.575   | 83.819   | 40     | 56     | 4      | 47.802  | 3973     | 1284     | 767      | 44     | 10   | 14   | 1    | 127    |
| MOT16-10 | 55.407 | 76.308 | 55.918 | 62.039   | 91.02    | 29.63  | 51.852 | 18.519 | 40.708  | 7642     | 4676     | 754      | 63     | 16   | 28   | 10   | 452    |
| MOT16-11 | 62.121 | 84.9   | 62.503 | 79.093   | 82.661   | 47.826 | 42.029 | 10.145 | 50.178  | 7256     | 1918     | 1522     | 35     | 33   | 29   | 7    | 159    |
| MOT16-13 | 41.921 | 74.593 | 42.594 | 49.59    | 87.637   | 22.43  | 43.925 | 33.645 | 29.322  | 5678     | 5772     | 801      | 77     | 24   | 47   | 36   | 279    |
| COMBINED | 54.137 | 77.461 | 54.569 | 64.673   | 84.487   | 32.108 | 47.389 | 20.503 | 39.676  | 71404    | 39003    | 11156    | 477    | 166  | 245  | 106  | 1953   |

IDF1: 57.992

  **Our Proposal**

|          | MOTA ⬆     | MOTP ⬆     | MODA ⬆     | CLR_Re ⬆ | CLR_Pr ⬆   | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆    | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇  | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇   |
| -------- | ---------- | ---------- | ---------- | -------- | ---------- | ------ | ------ | ------ | ---------- | -------- | -------- | -------- | ------- | ---- | ---- | ---- | -------- |
| MOT16-02 | 34.145     | 78.283     | 34.632     | 36.993   | 94.001     | 16.667 | 37.037 | 46.296 | 26.111     | 6597     | 11236    | 421      | 87      | 9    | 20   | 25   | 175      |
| MOT16-04 | 64.052     | 76.437     | 64.239     | 72.048   | 90.221     | 42.169 | 40.964 | 16.867 | 47.074     | 34264    | 13293    | 3714     | 89      | 35   | 34   | 14   | 381      |
| MOT16-05 | 59.431     | 78.371     | 60.384     | 69.845   | 88.071     | 29.6   | 57.6   | 12.8   | 44.324     | 4762     | 2056     | 645      | 65      | 37   | 72   | 16   | 169      |
| MOT16-09 | 61.936     | 84.136     | 62.831     | 75.423   | 85.693     | 52     | 44     | 4      | 49.972     | 3965     | 1292     | 662      | 47      | 13   | 11   | 1    | 84       |
| MOT16-10 | 54.571     | 76.877     | 55.066     | 59.328   | 93.298     | 25.926 | 50     | 24.074 | 40.852     | 7308     | 5010     | 525      | 61      | 14   | 27   | 13   | 346      |
| MOT16-11 | 66.067     | 85.202     | 66.372     | 78.286   | 86.792     | 49.275 | 34.783 | 15.942 | 54.482     | 7182     | 1992     | 1093     | 28      | 34   | 24   | 11   | 82       |
| MOT16-13 | 40.952     | 75.103     | 41.415     | 45.843   | 91.192     | 17.757 | 44.86  | 37.383 | 29.539     | 5249     | 6201     | 507      | 53      | 19   | 48   | 40   | 208      |
| COMBINED | **55.549** | **78.039** | **55.938** | 62.792   | **90.159** | 31.141 | 45.648 | 23.211 | **41.759** | 69327    | 41080    | **7567** | **430** | 161  | 236  | 120  | **1445** |

IDF1: **58.946**



## Thoughts After the Milestone

We need to reconsider the work critically. The generated images may help with the pre-training of the re-id backbone when the baseline is weak. But now the mAP could easily reach 0.9+, the simple GAN enhancement should not work well.

However, [AGW](https://github.com/mangye16/ReID-Survey) method is surprisingly NOT working as good as expected.
