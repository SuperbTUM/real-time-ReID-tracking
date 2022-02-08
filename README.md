## Update

- Feb. 8, 2022 Reformat files
- Feb. 4, 2022  Add some thoughts 
- Jan. 3, 2022  Add more details
- Dec. 20, 2021  First submit



## Introduction

This is a project of real-time multiple object tracking in person re-identification in collaboration with two data science master students. The proposal is to re-design the Re-ID model and try to obtain a stronger backbone. Several thoughts were applied and will be discussed throughly in the following chapters. The baseline is a Yolov5 based DeepSort algorithm and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch). The author has updated his repository with import of fast-reid package. I developed my code with Dec. 2021 version when the re-id model from scratch existed. The baseline backbone was ResNet-18.



## Dataset

We use MOT16 as benchmark and Market1501 to train our re-id network.

Video: [MOT16](https://motchallenge.net/data/MOT16/) --> This dataset could be evaluated with completed bash scripts.

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data) --> We only use the training set for training. (It is expected to use all the images but we just want to have 751 classes.)



## Quick Start

Please do not use Google Colab, since file transfer is rather unstable!

Replace the file with modified one in the original repositories.

You can try to have generated images with GAN. That means you need additional training on the GAN. To train DC-GAN, please refer to the instructions on [qiaoguan's repo](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). Please pay attention that this is an out-of-state repo and there is also file missing. You can also refer to the `ipynb` file in [modification_dcgan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_dcgan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py` before conducting training. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

You will need to train the Re-ID model with Market1501. In the [modification_deepsort](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort) folder, you can see how we build the model as well train the model. The intuition is to use bag of tricks. We firstly trained the embedding network and classification network separately, and then we merged them together with merged loss. We have our checkpoint available [here](https://drive.google.com/file/d/1dP3afrkTWyYlLOGJFbIi4QWIxKoPYjOl/view?usp=sharing). 

For the final evaluation, please refer to [mikel-brostrom's repo](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) for details. We also have our script file for tracking (May not reliable as running in Colab).



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The inference speed is 15 ~ 17 ms per frame with 640 * 640 resolution with Tesla T4. The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 17 ms per frame under the same testing environment. The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. The evaluation can be deployed with a bash command. I set the maximum distance to 0.15 (cosine distance). The evaluation results are shown below. Note that this is currently a **NOT** very successful attempt as the baseline is too strong.

**Strong baseline**

|          | MOTA   | MOTP   | MODA   | CLR_Re | CLR_Pr | MTR    | PTR    | MLR    | sMOTA  | CLR_TP | CLR_FN | CLR_FP | IDSW | MT   | PT   | ML   | Frag |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- |
| MOT16-02 | 35.188 | 77.36  | 35.748 | 40.279 | 89.889 | 16.667 | 42.593 | 40.741 | 26.068 | 7183   | 10650  | 808    | 100  | 9    | 23   | 22   | 287  |
| MOT16-04 | 61.171 | 76.199 | 61.362 | 73.379 | 85.928 | 42.169 | 42.169 | 15.663 | 43.706 | 34897  | 12660  | 5715   | 91   | 35   | 35   | 13   | 468  |
| MOT16-05 | 57.48  | 78.314 | 58.463 | 70.035 | 85.82  | 31.2   | 55.2   | 13.6   | 42.293 | 4775   | 2043   | 789    | 67   | 39   | 69   | 17   | 181  |
| MOT16-09 | 60.148 | 83.663 | 60.985 | 75.575 | 83.819 | 40     | 56     | 4      | 47.802 | 3973   | 1284   | 767    | 44   | 10   | 14   | 1    | 127  |
| MOT16-10 | 55.407 | 76.308 | 55.918 | 62.039 | 91.02  | 29.63  | 51.852 | 18.519 | 40.708 | 7642   | 4676   | 754    | 63   | 16   | 28   | 10   | 452  |
| MOT16-11 | 62.121 | 84.9   | 62.503 | 79.093 | 82.661 | 47.826 | 42.029 | 10.145 | 50.178 | 7256   | 1918   | 1522   | 35   | 33   | 29   | 7    | 159  |
| MOT16-13 | 41.921 | 74.593 | 42.594 | 49.59  | 87.637 | 22.43  | 43.925 | 33.645 | 29.322 | 5678   | 5772   | 801    | 77   | 24   | 47   | 36   | 279  |
| COMBINED | 54.137 | 77.461 | 54.569 | 64.673 | 84.487 | 32.108 | 47.389 | 20.503 | 39.676 | 71404  | 39003  | 11156  | 477  | 166  | 245  | 106  | 1953 |

  **Current attempt**

|          | MOTA   | MOTP      | MODA   | CLR_Re | CLR_Pr | MTR    | PTR    | MLR    | sMOTA  | CLR_TP | CLR_FN | CLR_FP | IDSW | MT   | PT   | ML   | Frag |
| -------- | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- |
| MOT16-02 | 34.649 | 77.546    | 35.371 | 39.908 | 89.963 | 16.667 | 42.593 | 40.741 | 25.711 | 7099   | 10734  | 792    | 128  | 9    | 23   | 22   | 280  |
| MOT16-04 | 61.24  | 76.122    | 61.436 | 73.358 | 86.02  | 40.964 | 44.578 | 14.458 | 43.723 | 34887  | 12670  | 5670   | 93   | 34   | 37   | 12   | 472  |
| MOT16-05 | 56.688 | 78.356    | 58.038 | 69.346 | 85.979 | 30.4   | 55.2   | 14.4   | 41.679 | 4728   | 2090   | 771    | 92   | 38   | 69   | 18   | 177  |
| MOT16-09 | 59.958 | 84.195    | 61.042 | 75.005 | 84.306 | 52     | 44     | 4      | 48.103 | 3943   | 1314   | 734    | 57   | 13   | 11   | 1    | 107  |
| MOT16-10 | 55.326 | 76.365    | 55.894 | 61.926 | 91.124 | 27.778 | 53.704 | 18.519 | 40.689 | 7628   | 4690   | 743    | 70   | 15   | 29   | 10   | 446  |
| MOT16-11 | 62.753 | 84.964    | 63.146 | 79.06  | 83.243 | 47.826 | 40.58  | 11.594 | 50.866 | 7253   | 1921   | 1460   | 36   | 33   | 28   | 8    | 136  |
| MOT16-13 | 41.581 | 74.652    | 42.279 | 49.275 | 87.568 | 21.495 | 46.729 | 31.776 | 29.091 | 5642   | 5808   | 801    | 80   | 23   | 50   | 34   | 268  |
| COMBINED | 54.03  | **77.67** | 54.534 | 64.471 | 86.645 | 31.915 | 47.776 | 20.309 | 39.634 | 71180  | 39227  | 10971  | 556  | 165  | 247  | 105  | 1876 |



## Thoughts After the Milestone

We need to reconsider the work critically. The generated images may help with the pre-training of the re-id backbone when the baseline is weak. But now the mAP could easily reach 0.9+, the GAN enhancement should not work well.

The training of the re-id backbone is naive. Still I want a light backbone, but it seems that it is hard to optimize ResNet18. Since the BagOfTricks method is NOT working as good as expected, I am switching to [AGW](https://github.com/mangye16/ReID-Survey) method.
