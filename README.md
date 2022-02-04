## Update

- Feb. 4, 2022  Add some thoughts 
- Jan. 3, 2022  Add more details
- Dec. 20, 2021  First submit



## Introduction

This is a project of real-time multiple object tracking in person re-identification. The proposal is to re-design the Re-ID model by having a separable generative and discriminative network. The baseline is a Yolov5 based DeepSort tracker and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).



## Dataset

We use MOT16 as benchmark and Market1501 to train our re-id network.

Video: [MOT16](https://motchallenge.net/data/MOT16/)

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data)



## Quick Start

Please do not use Google Colab!

To train DC-GAN, please refer to the instructions on [qiaoguan's repo](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). You can also refer to the `ipynb` file in [modification_dcgan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_dcgan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py`. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

In the [modification_deepsort](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort) folder, you can see how we build the model as well train the model. We have our checkpoint available [here](https://drive.google.com/file/d/1jS6m1f52MWx8_Gd_Gk55qVTCeDHb_Zhz/view?usp=sharing). 

For the final evaluation, please refer to [mikel-brostrom's repo](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) for details. We also have our script file for tracking.



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The tracking speed is 1.5 ms per frame with 640 * 640 resolution with Tesla P100. The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 1.7 ms per frame under the same testing environment.



## Evaluation

The tracking quality is evaluated under MOTA and ID-Switches. The evaluation can be deployed with a bash command.



## Thoughts After the Milestone

We need to reconsider the optimization critically. The generated images may help with the pre-training of the re-id backbone when the baseline is weak. But now the mAP could easily reach 0.9+, the GAN enhancement should not work well.

The training of the re-id backbone is naive. It builds a classification network while it needs to classify hundreds kinds of objects with a tiny network. This makes the classification unfeasible and the features may be unreliable. We can train the network using contrastive loss. It considers the difference between two objects, in other words, if the two objects have the same labels, they should be similar in distance, otherwise, they should be far away from each other. However, there is another issue regarding the pose of the object. The network struggles to match the same objects with either front or back. We leveraged OpenPose from CMU to extract the poses and enriched the loss function by taking both similarity and poses into account. Have a brief look at the demo [here](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort). Note that the selection of margin in contrastive loss and weights assignment are tricky.
