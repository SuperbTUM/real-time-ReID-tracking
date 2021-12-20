## Introduction

This is a project of real-time multiple object tracking in person re-identification. The proposal is to re-design the Re-ID model by having a separable generative and discriminative network. The baseline is a Yolov5 based DeepSort tracker and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).



## Dataset

Video: [MOT16](https://motchallenge.net/data/MOT16/)

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data)



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The tracking speed is 1.5 ms per frame with 640 * 640 resolution with Tesla P100. The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 1.7 ms per frame under the same testing environment.



## Evaluation

The tracking quality is evaluated under MOTA and ID-Switches. 

