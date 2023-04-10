## TODOs

- [x] To work with latest YoloV8-DeepOCSort implementation towards a new benchmark(important!) and apply necessary changes on current repository.

- [x] To debug and tune a VAE-(W)GAN model for Market1501.

- [ ] To check if triplet loss w/. penalty is helpful in object reidentification/image retrivial/person search.

- [ ] To work with sequential side information on ViT and Swin-transformer.

- [x] To check whether background information is helpful with a hyperparameter to zoom, inspired by an ECCV18' paper.

  

## Update

- Mar. 15, 2023, Add Swin-Transformer & ViT w/. Side Information and multi-scale features
- Jan. 24, 2023, Add VAE-GAN/WGAN to support generating better synthetic images
- Sep. 24, Support distributed training w/. PyTorch
- Feb. 27, Consider class-imbalanced problem and build balanced dataset for Market-1501 training set
- Feb. 22, Add Video-based training code and switch IoU to DIoU
- Feb. 11, Add IBN and modified MIN_CONF to 0.5
- Feb. 8, 2022 Reformat files
- Feb. 4, 2022  Add some thoughts 
- Jan. 3, 2022  Add more details
- Dec. 20, 2021  First submit



## Introduction

This is a project of real-time multiple object tracking in person re-identification. The proposal is to re-design the Re-ID model and try to obtain a stronger backbone. Several thoughts were applied and will be discussed throughly in the following chapters. 

The baseline is a Yolov5(now is YoloV8!) based DeepSort(now is DeepOCSort!) algorithm and can be found [here](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch). The author has updated his repository with import of fast-reid package. I developed my code with Dec. 2021 version when the re-id model from scratch existed. The baseline backbone was ResNet-18(now is OSNet!).



## Dependency

Python >= 3.8 is recommended.

All code was developed with PyTorch and Numpy. Please download and install the latest stable version with CUDA 11.x. All codes are tested with one single GPU. If you wish to accelerate training, [mixed-precision training](https://github.com/NVIDIA/apex) should be a great option. It can save training time as well as reduce memory cost. Distributed training is also available. If you wish to accelerate inference, you may refer to TorchScript or TensorRT. A major reference is [this](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/).

You are suggested to install other dependencies with

```bash
pip install -r requirements.txt
```



## Dataset

We use MOT16 evaluation as benchmark and Market1501 to train our re-id network.

Videos: [MOT16](https://motchallenge.net/data/MOT16/) => This dataset could be evaluated with completed bash scripts.

Person gallery: [Market1501](https://www.kaggle.com/pengcw1/market-1501/data) => We only use the training set for training. (It is expected to use all the images but we just want to have 751 classes.)



## Quick Start

[Pending] Replace the file with modified one in the original repositories.

**GAN training**

You can try to have generated images with GAN. That means you need additional training on the GAN. We have a few selections: DC-GAN, VAE-GAN and VAE-WGANGP. Explicitly, to train DC-GAN, please refer to the instructions on [this](https://github.com/qiaoguan/Person-reid-GAN-pytorch/tree/master/DCGAN-tensorflow). Please pay attention that this is an out-of-fashion repo w/. certain file missing. You can also refer to the `ipynb` file in [modification_gan](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_gan) folder. Before that, make sure you properly execute `prepare.py`, `changeIndex.py` as well as the customized `re_index.py` before conducting training. We trained the backbone with `Market-1501/bounding_box_train` plus generated images. You can refer to the script files in our repo. 

In a general scenario, you can simply execute the training script:

```python
python modification_gan/synthetic_generate.py --vae --Wassertein --gp
```

**ReID training**

You will need to train your Re-ID model with Market1501. In the [reid](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/reid) folder, you can see how we build the model as well train the model. There are a few versions of models.

The intuition is to apply bag of tricks. We considered random erasing augmentation, last stride reduction, center loss, SE block, batch norm neck, etc. We found that IBN and GeM modules are critical, so we also include this in our backbone. We have our checkpoint available [here](https://drive.google.com/file/d/1Ta89D7WXhL_H2lR_eYEyLuWfZEFdNEXo/view?usp=sharing). 

We also have plr_osnet, vision transformer from Trans-ReID, and swin transformer.

We train the model on both image-based dataset and video-based dataset(w/. ground truth), and the scripts can be access under the same folder.

```python
python reid/image_reid_train.py --backbone vit --accelerate
```

```python
python reid/video_reid_train.py --crop_factor 1.0
```

To fit with `yolov8_tracking`, please copy your model and checkpoints to `trackers/strongsort/models` and `trackers/strongsort/checkpoint`, and modify `reid_model_factory.py` accordingly.

If you want to train the Re-ID model with video dataset, please refer to the [video_train](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/modification_deepsort/mot16_train.py) script.

**Tracking evaluation**

For the final evaluation, please refer to [this Wiki](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation) for details. We also have our script file for tracking (May not reliable as running in Colab).



## Tracking Speed

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. The inference speed is 15 ~ 20 ms per frame [need to be re-assessed and OSNet 1.0 is heavy!!! ~100ms per frame] depending on the sparsity of pedestrians with 640 * 640 resolution with Tesla P100. It may be slower if bounding box is resized to (128, 256). The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. The tracking speed is 17 ms per frame under the same testing environment. The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

**Re-identification**

SeRes18-IBN-BatchReNorm(Important!)-PolyLoss(Important; epsilon=1.0)+SoftTriplet+Center

[Checkpoint](https://drive.google.com/file/d/1iiCiGgkWMUlgAiqoGNsWKpasXtbmNkjd/view?usp=share_link)

| Metric | Acc@1  | Acc@5  | Acc@10 | mAP    |
| ------ |--------|--------|--------|--------|
| Value  | 0.7587 | 0.9062 | 0.9489 | 0.5195 |

SeRes18-IBN-BatchReNorm(Important!)-Softmax+SoftTriplet+Center

| Metric | Acc@1  | Acc@5  | Acc@10 | mAP    |
| ------ |--------|--------|--------|--------|
| Value  | 0.7304 | 0.8863 | 0.9353 | 0.4954 |



**Tracking**

[Need to be re-evaluated]

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. The evaluation can be deployed with a bash command. I set the maximum distance to 0.15 (cosine distance) and minimum confidence to 0.5. Also, I resized the bounding box to (128, 256). The evaluation results are shown below. Our proposal has better performance in MOTA, MOTP, MODA and IDF1 with 1% of absolute improvement! For specific meaning of tracking metrics, please refer to [this](https://link.springer.com/content/pdf/10.1007/s11263-020-01375-2.pdf) and [this](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf).

**Original Baseline (YoloV8m detector)**

```bash
python val.py --benchmark MOT16 --tracking-method strongsort --device 0 --img 640 --processes-per-device 4 --yolo-weights /home/mh4116/yolov8_tracking/crowdhuman_yolov5m.pt
```

YoloV8 COCO detector is weak.

|          | MOTA ⬆ | MOTP ⬆ | MODA ⬆ | CLR_Re ⬆ | CLR_Pr ⬆ | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆ | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇ | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇ |
| -------- | ------ | ------ | ------ | -------- | -------- | ------ | ------ | ------ | ------- | -------- | -------- | -------- | ------ | ---- | ---- | ---- | ------ |
| MOT16-02 | 21.881 | 85.498 | 21.999 | 22.711   | 96.96    | 12.963 | 24.074 | 62.963 | 18.587  | 4050     | 13783    | 127      | 21     | 7    | 13   | 34   | 75     |
| MOT16-04 | 28.917 | 85.24  | 28.974 | 30.832   | 94.314   | 8.4337 | 39.759 | 51.807 | 24.366  | 14663    | 32894    | 884      | 27     | 7    | 33   | 43   | 238    |
| MOT16-05 | 52.156 | 77.17  | 52.728 | 61.426   | 87.597   | 20     | 58.4   | 21.6   | 38.133  | 4188     | 2630     | 593      | 39     | 25   | 73   | 27   | 131    |
| MOT16-09 | 64.809 | 81.893 | 65.456 | 70.287   | 93.568   | 48     | 40     | 12     | 52.082  | 3695     | 1562     | 254      | 34     | 12   | 10   | 3    | 59     |
| MOT16-10 | 40.372 | 78.124 | 40.705 | 44.756   | 91.7     | 18.519 | 35.185 | 46.296 | 30.581  | 5513     | 6805     | 499      | 41     | 10   | 19   | 25   | 141    |
| MOT16-11 | 53.107 | 85.274 | 53.423 | 64.16    | 85.664   | 28.986 | 34.783 | 36.232 | 43.659  | 5886     | 3288     | 985      | 29     | 20   | 24   | 25   | 45     |
| MOT16-13 | 25.852 | 79.639 | 26.131 | 28.306   | 92.865   | 10.28  | 33.645 | 56.075 | 20.088  | 3241     | 8209     | 249      | 32     | 11   | 36   | 60   | 120    |
| COMBINED | 33.895 | 82.759 | 34.097 | 37.349   | 91.989   | 17.795 | 40.232 | 41.973 | 27.455  | 41236    | 69171    | 3591     | 223    | 92   | 208  | 217  | 809    |

YoloV5 detector is not working right now...

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

  **Our Proposal w/o Data Balance**

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

**Our Proposal w/ Data Balance**, [checkpoint](https://drive.google.com/file/d/1-NKAC0__QOK28zKXD0dqiv6pihlCcZLK/view?usp=sharing)

|          | MOTA ⬆ | MOTP ⬆ | MODA ⬆ | CLR_Re ⬆ | CLR_Pr ⬆ | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆ | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇ | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇ |
| -------- | ------ | ------ | ------ | -------- | -------- | ------ | ------ | ------ | ------- | -------- | -------- | -------- | ------ | ---- | ---- | ---- | ------ |
| MOT16-02 | 33.898 | 78.3   | 34.419 | 36.859   | 93.793   | 16.667 | 38.889 | 44.444 | 25.899  | 6573     | 11260    | 435      | 93     | 9    | 21   | 24   | 178    |
| MOT16-04 | 64.052 | 76.393 | 64.256 | 72.067   | 90.221   | 42.169 | 40.964 | 16.867 | 47.039  | 34273    | 13284    | 3715     | 97     | 35   | 34   | 14   | 387    |
| MOT16-05 | 58.903 | 78.458 | 59.724 | 69.317   | 87.844   | 28.8   | 56.8   | 14.4   | 43.971  | 4726     | 2092     | 654      | 56     | 36   | 71   | 18   | 159    |
| MOT16-09 | 62.317 | 84.061 | 63.116 | 75.594   | 85.832   | 52     | 44     | 4      | 50.268  | 3974     | 1283     | 656      | 42     | 13   | 11   | 1    | 88     |
| MOT16-10 | 54.522 | 76.896 | 55.041 | 59.336   | 93.251   | 25.926 | 50     | 24.074 | 40.813  | 7309     | 5009     | 529      | 64     | 14   | 27   | 13   | 339    |
| MOT16-11 | 66.514 | 85.207 | 66.819 | 78.363   | 87.161   | 49.275 | 34.783 | 15.942 | 54.922  | 7189     | 1985     | 1059     | 28     | 34   | 24   | 11   | 77     |
| MOT16-13 | 40.672 | 75.058 | 41.162 | 45.721   | 90.933   | 18.692 | 44.86  | 36.449 | 29.269  | 5235     | 6215     | 522      | 56     | 20   | 48   | 39   | 204    |
| COMBINED | 55.497 | 78.022 | 55.892 | 62.749   | 90.15    | 31.141 | 45.648 | 23.211 | 41.706  | 69279    | 41128    | 7570     | 436    | 161  | 236  | 120  | 1432   |

IDF1: **59.31**



## Thoughts After the Milestone

Simple DCGAN may bring easy samples to make the network more likely to overfit. We may pre-cluster (K-Means++) datasets and conduct synthetic image generation on each cluster with stronger GAN network (VAE-GAN). We can even make the network partial-aware with separate attention on foreground and background.
