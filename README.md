## Update

- Feb. 11, Add IBN and modified MIN_CONF to 0.5
- Feb. 8, 2022 Reformat files
- Feb. 4, 2022  Add some thoughts 
- Jan. 3, 2022  Add more details
- Dec. 20, 2021  First submit



## Introduction

This is a project of real-time multiple object tracking in person re-identification in collaboration with two data science master students. The proposal is to re-design the Re-ID model and try to obtain a stronger backbone. Several thoughts were applied and will be discussed throughly in the following chapters. The baseline is a Yolov5 based DeepSort algorithm and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch). The author has updated his repository with import of fast-reid package. I developed my code with Dec. 2021 version when the re-id model from scratch existed. The baseline backbone was ResNet-18.



## Dataset

We use MOT16 as benchmark and Market1501 to train our re-id network.

Videos: [MOT16](https://motchallenge.net/data/MOT16/) => This dataset could be evaluated with completed bash scripts.

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data) => We only use the training set for training. (It is expected to use all the images but we just want to have 751 classes.)



## Quick Start

Please do not use Google Colab, even if you have a Pro account, since file transfer is rather unstable (especially in the evaluation stage)!

Replace the file with modified one in the original repositories.

You can try to have generated images with GAN. That means you need additional training on the GAN. To train DC-GAN, please refer to the instructions on [this](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). Please pay attention that this is an out-of-state repo and there is also file missing. You can also refer to the `ipynb` file in [modification_dcgan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_dcgan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py` before conducting training. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

You will need to train the Re-ID model with Market1501. In the [modification_deepsort](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort) folder, you can see how we build the model as well train the model. The intuition is to use bag of tricks. We considered random erasing augmentation, last stride reduction, center loss, SE block, batch norm neck, etc. We have our checkpoint available [here](https://drive.google.com/file/d/1Ta89D7WXhL_H2lR_eYEyLuWfZEFdNEXo/view?usp=sharing). 

For the final evaluation, please refer to [this Wiki](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation) for details. We also have our script file for tracking (May not reliable as running in Colab).



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The inference speed is 15 ~ 20 ms per frame depending on the sparsity of pedestrians with 640 * 640 resolution with Tesla T4. The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 17 ms per frame under the same testing environment. The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. The evaluation can be deployed with a bash command. I set the maximum distance to 0.15 (cosine distance) and minimum confidence to 0.5. The evaluation results are shown below. Our proposal has better performance in MOTA, MOTP and MODA with 1% of absolute improvement! For specific meaning of tracking metrics, please refer to [this](https://link.springer.com/content/pdf/10.1007/s11263-020-01375-2.pdf) and [this](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf).

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

  **Our Proposal**

|          | MOTA ⬆     | MOTP ⬆     | MODA ⬆     | CLR_Re ⬆ | CLR_Pr ⬆   | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆    | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇ | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇   |
| -------- | ---------- | ---------- | ---------- | -------- | ---------- | ------ | ------ | ------ | ---------- | -------- | -------- | -------- | ------ | ---- | ---- | ---- | -------- |
| MOT16-02 | 33.556     | 78.416     | 34.133     | 36.399   | 94.141     | 16.667 | 37.037 | 46.296 | 25.7       | 6491     | 11342    | 404      | 103    | 9    | 20   | 25   | 162      |
| MOT16-04 | 64.045     | 76.464     | 64.23      | 71.96    | 90.3       | 40.964 | 42.169 | 16.867 | 47.109     | 34222    | 13335    | 3676     | 88     | 34   | 35   | 14   | 368      |
| MOT16-05 | 57.583     | 78.511     | 58.786     | 68.084   | 87.983     | 27.2   | 56.8   | 16     | 42.952     | 4642     | 2176     | 634      | 82     | 34   | 71   | 20   | 148      |
| MOT16-09 | 61.537     | 84.286     | 62.716     | 74.796   | 86.096     | 52     | 44     | 4      | 49.784     | 3932     | 1325     | 635      | 62     | 13   | 11   | 1    | 78       |
| MOT16-10 | 54.433     | 76.988     | 54.952     | 58.881   | 93.744     | 27.778 | 48.148 | 24.074 | 40.883     | 7253     | 5065     | 484      | 64     | 15   | 26   | 13   | 324      |
| MOT16-11 | 66.569     | 85.247     | 66.896     | 78.101   | 87.453     | 50.725 | 33.333 | 15.942 | 55.046     | 7165     | 2009     | 1028     | 30     | 35   | 23   | 11   | 80       |
| MOT16-13 | 40.507     | 75.258     | 41.092     | 45.074   | 91.882     | 16.822 | 44.86  | 38.318 | 29.354     | 5161     | 6289     | 456      | 67     | 18   | 48   | 41   | 177      |
| COMBINED | **55.298** | **78.111** | **55.747** | 62.375   | **90.395** | 30.561 | 45.261 | 24.178 | **41.645** | 68866    | 41541    | **7317** | 496    | 158  | 234  | 125  | **1337** |



## Thoughts After the Milestone

We need to reconsider the work critically. The generated images may help with the pre-training of the re-id backbone when the baseline is weak. But now the mAP could easily reach 0.9+, the simple GAN enhancement should not work well.

However, [AGW](https://github.com/mangye16/ReID-Survey) method is surprisingly NOT working as good as expected.
