## Update

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

You can try to have generated images with GAN. That means you need additional training on the GAN. To train DC-GAN, please refer to the instructions on [qiaoguan's repo](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). Please pay attention that this is an out-of-state repo and there is also file missing. You can also refer to the `ipynb` file in [modification_dcgan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_dcgan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py` before conducting training. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

You will need to train the Re-ID model with Market1501. In the [modification_deepsort](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort) folder, you can see how we build the model as well train the model. The intuition is to use bag of tricks. We firstly trained the embedding network and classification network separately, and then we merged them together with merged loss. We have our checkpoint available [here](https://drive.google.com/file/d/1dP3afrkTWyYlLOGJFbIi4QWIxKoPYjOl/view?usp=sharing). 

For the final evaluation, please refer to [mikel-brostrom's repo](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) for details. We also have our script file for tracking (May not reliable as running in Colab).



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The tracking speed is 15 ms per frame with 640 * 640 resolution with Tesla P100. The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 17 ms per frame under the same testing environment. The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. The evaluation can be deployed with a bash command. The evaluation results are shown below. Note that this is currently a **not** very successful attempt as the baseline is too strong.

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

|          | MOTA   | MOTP   | MODA   | CLR_Re | CLR_Pr | MTR    | PTR    | MLR    | sMOTA  | CLR_TP | CLR_FN | CLR_FP | IDSW | MT   | PT   | ML   | Frag |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- |
| MOT16-02 | 34.644 | 77.466 | 35.361 | 39.926 | 89.74  | 16.667 | 42.593 | 40.741 | 25.647 | 7120   | 10713  | 814    | 128  | 9    | 23   | 22   | 280  |
| MOT16-04 | 61.204 | 76.227 | 61.375 | 73.534 | 85.962 | 40.964 | 43.373 | 15.663 | 43.766 | 34885  | 12672  | 5697   | 81   | 34   | 36   | 13   | 474  |
| MOT16-05 | 56.762 | 78.421 | 58.111 | 69.346 | 86.058 | 29.6   | 57.6   | 12.8   | 41.797 | 4728   | 2090   | 766    | 92   | 37   | 72   | 16   | 178  |
| MOT16-09 | 59.635 | 84.116 | 60.89  | 74.986 | 84.177 | 52     | 44     | 4      | 47.724 | 3942   | 1315   | 741    | 66   | 13   | 11   | 1    | 118  |
| MOT16-10 | 54.863 | 76.448 | 55.488 | 61.552 | 91.031 | 27.778 | 53.704 | 18.519 | 40.366 | 7582   | 4736   | 747    | 77   | 15   | 29   | 10   | 441  |
| MOT16-11 | 62.525 | 84.973 | 62.906 | 78.984 | 83.087 | 49.275 | 39.13  | 11.594 | 50.656 | 7246   | 1928   | 1475   | 35   | 34   | 27   | 8    | 140  |
| MOT16-13 | 41.808 | 74.654 | 42.541 | 49.319 | 87.918 | 22.43  | 44.86  | 32.71  | 29.308 | 5647   | 5803   | 776    | 84   | 24   | 48   | 35   | 256  |
| COMBINED | 53.956 | 77.723 | 54.466 | 64.443 | 86.593 | 32.108 | 47.582 | 20.309 | 39.67  | 1150   | 39257  | 11016  | 563  | 166  | 246  | 105  | 1887 |



## Thoughts After the Milestone

We need to reconsider the work critically. The generated images may help with the pre-training of the re-id backbone when the baseline is weak. But now the mAP could easily reach 0.9+, the GAN enhancement should not work well.

The training of the re-id backbone is naive. Still I want a light backbone. Without re-ranking, the classification accuracy for training set reached 78.8% in 25 epochs in merged models and ~72% in separated models in 15 epochs respectively. I checked the tricks of SE block as well as NLA (non-local attention) and don't know which one is better. Note that the loss weight assignment and margin setting are tricky.
