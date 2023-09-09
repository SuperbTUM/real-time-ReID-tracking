# Real-time ReID Tracking w/. Strong Feature Extractor & GAN
[Slides](https://docs.google.com/presentation/d/1cs5wEIcJWV5H6F5xnWDV0ikYf-uWuFvZj3byTBs4sMQ/edit#slide=id.p), Manuscript(Coming Soon...)

## Introduction

From Sort to OCSort, we are becoming aware that deep feature extractor is crucial in both re-identification and multiple object tracking.
The project integrates Yolo detection, GAN, deep feature extractor for re-identification and MOT. 
The baseline is a Yolov5(now is YoloV8!) based DeepSort(now is DeepOCSort & StrongSort) algorithm.
Everything can be found [here](https://github.com/mikel-brostrom/yolo_tracking). 
The author has updated the repository with import of fast-reid package.  
The baseline backbone for re-identification is simply ResNet-18.



## Dependency

Python >= 3.8 is recommended.

All code was developed with latest version of PyTorch(2.0!) and Numpy. 
Please download and install the latest stable version with CUDA 11.x(ex. 11.8). 
All codes are tested with one single GPU. 
If you wish to accelerate training, you are advised to apply [mixed-precision training](https://github.com/NVIDIA/apex). 
It is estimated to save training time as well as reduce memory cost. 
Distributed training is also available. 
If you wish to accelerate inference, you may refer to TorchScript, ONNX or TensorRT. 
A major reference is [this](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/).

You are suggested to install other dependencies with

```bash
pip --disable-pip-version-check install -r requirements.txt
```

With >= Python3.9, you can install CUML with
```bash
pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

Optionally, using `faiss` is highly recommended.

```bash
conda install -c conda-forge faiss-gpu
conda install faiss-gpu=1.7.4
```


## Datasets

The current datasets are a bit outdated. You are advised to use MOT17, MOT20 instead.
Now we primarily use MOT16 evaluation as benchmark and Market1501/DukeMTMC to pre-train our re-id network.

Videos: [MOT16](https://motchallenge.net/data/MOT16/)

Person gallery: [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ), [DukeMTMC-reID](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view)



## Quick Start

[Pending] Replace the file with modified one in the original repositories.

**GAN training**

You can try to generate synthetic images with GAN. 
That means you need additional training on the GAN. 
We have a few pending selections: DC-GAN, VAE-GAN and VAE-WGANGP. 
There are some references([ref1](https://arxiv.org/abs/1805.08318), [ref2](https://arxiv.org/abs/1802.05957)) for robust training.
Unfortunately we don't have diffusion model at this moment.

In a general scenario, you can simply execute the training script:

```python
python modification_gan/synthetic_generate.py --ngf 256 --ndf 64 --ema
```

**ReID training**

Although some checkpoints are available, you are still advised to train your Re-ID model with Market1501.
Due to privacy issue, some datasets such as DukeMTMC are no longer open to the public and not acceptable to the academy as well.
In the [reid](https://github.com/SuperbTUM/real-time-ReID-tracking/tree/main/reid) folder, you can see how we build the model as well train the model. 
There are a few versions of models.

The main focus of the project is to construct a lite backbone for mobile development and real-time tracking. 
But still, we include a model zoo, with CNN-based re-id models, and vision transformer based models, where you can access all of them in the [backbones](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/reid/backbones) folder.
We train the model on both image-based dataset and video-based dataset(w/. ground truth), and the scripts can be access under the same folder.

For non-continual image training
```python
python reid/image_reid_train.py --bs 32 --backbone cares18 --accelerate --renorm --epsilon -1.0 --margin 0.3 --center_lamda 0.0005 --instance 8
```

For continual image training
```python
python reid/image_reid_train.py --bs 32 --backbone cares18 --accelerate --renorm --epsilon -1.0 --margin 0.3 --center_lamda 0.0005 --instance 8 --continual --eps 0.5
```

For image testing (and we are about to support dynamic batching!)
```python
python reid/image_reid_inference.py --backbone cares18 --bs 1 --ckpt checkpoint/reid_model.onnx --eps 0.5
```
Or in a faster way
```python
python reid/image_reid_inference.py --backbone cares18 --bs 16 --ckpt checkpoint/reid_model_xxx.onnx --eps 0.5
```

For video training
```python
python reid/video_reid_train.py --crop_factor 1.0
```

Training with `accelerate`
```python
accelerate config
CUDA_VISIBLE_DEVICES="0" accelerate launch xxx.py --args
```

To fit with `yolov8_tracking`, please copy your model and checkpoints to `trackers/strongsort/models` and `trackers/strongsort/checkpoint`, and modify `reid_model_factory.py` accordingly.
If you want to train the Re-ID model with video dataset, please refer to the [video_train](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/reid/video_reid_train.py) script.

**Tracking evaluation**

For the final evaluation, please refer to [this Wiki](https://github.com/mikel-brostrom/yolo_tracking/wiki/MOT-16-evaluation) for details. 
We also have our script file for tracking.



## Tracking Speed

[The following conclusion is outdated]

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. 
The inference speed is 15 ~ 20 ms per frame [need to be re-assessed and OSNet 1.0 is heavy!!! ~100ms per frame] depending on the sparsity of pedestrians with 640 * 640 resolution with Tesla P100. 
It may be slower if bounding box is resized to (128, 256). 
The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. 
The tracking speed is 17 ms per frame under the same testing environment. 
The speed is acquired with `time` package after the synchronization of CUDA.



## Evaluation

**Re-identification**
[Checkpoints](https://drive.google.com/drive/folders/1GC8CXXfMbfCR_-hJaMKGTNUx2-XU9Gp4?usp=sharing)

* Metric-learning

Market1501

| Continual | Metric    | Acc@1  | mAP    |
|-----------|-----------|--------|--------|
| []        | Euclidean | 0.9240 | 0.7915 |
| []        | Jaccard   | 0.9267 | 0.8852 |
| [x]       | Jaccard   | 0.9463 | 0.9210 |


DukeMTMC

| Continual | Metric    | Acc@1  | mAP    |
|-----------|-----------|--------|--------|
| []        | Euclidean | 0.8182 | 0.6967 |
| []        | Jaccard   | 0.8366 | 0.7918 |
| [x]       | Jaccard   | 0.8757 | 0.8312 |

* Domain Transfer

Market1501 -> DukeMTMC

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| []        | Cross-domain | 0.4969 | 0.3334 |
| [x]       | Cross-domain | 0.5368 | 0.3896 |

DukeMTMC -> Market1501

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| []        | Cross-domain | 0.6256 | 0.3933 |
| [x]       | Cross-domain | 0.6580 | 0.4130 |

**Tracking**

[Need to be re-evaluated]

The tracking quality is evaluated under regular metrics including MOTA, MOTP and IDSW. HOTA will be applied soon.
The evaluation can be deployed with a bash command. 
I resized the bounding box to (128, 256), set the maximum distance to 0.15 (cosine distance) and minimum confidence to 0.5 for your information. 
The evaluation results are shown below. 
Our proposal has better performance in MOTA, MOTP, MODA and IDF1 with 1% of absolute improvement! 
For specific meaning of tracking metrics, please refer to [this](https://link.springer.com/content/pdf/10.1007/s11263-020-01375-2.pdf) and [this](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf).

**Original Baseline (YoloV8m detector)**

```bash
python val.py --benchmark MOT16 --tracking-method strongsort --device 0 --img 640 --processes-per-device 4 --yolo-weights /home/mh4116/yolov8_tracking/crowdhuman_yolov5m.pt
```

YoloV8 COCO detector is weak.

|          | MOTA ⬆ | MOTP ⬆ | MODA ⬆ | CLR_Re ⬆ | CLR_Pr ⬆ | MTR ⬆  | PTR ⬆  | MLR ⬇  | sMOTA ⬆ | CLR_TP ⬆ | CLR_FN ⬇ | CLR_FP ⬇ | IDSW ⬇ | MT ⬆ | PT ⬆ | ML ⬇ | Frag ⬇ |
|----------|--------|--------|--------|----------|----------|--------|--------|--------|---------|----------|----------|----------|--------|------|------|------|--------|
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
|----------|--------|--------|--------|----------|----------|--------|--------|--------|---------|----------|----------|----------|--------|------|------|------|--------|
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
|----------|------------|------------|------------|----------|------------|--------|--------|--------|------------|----------|----------|----------|---------|------|------|------|----------|
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
|----------|--------|--------|--------|----------|----------|--------|--------|--------|---------|----------|----------|----------|--------|------|------|------|--------|
| MOT16-02 | 33.898 | 78.3   | 34.419 | 36.859   | 93.793   | 16.667 | 38.889 | 44.444 | 25.899  | 6573     | 11260    | 435      | 93     | 9    | 21   | 24   | 178    |
| MOT16-04 | 64.052 | 76.393 | 64.256 | 72.067   | 90.221   | 42.169 | 40.964 | 16.867 | 47.039  | 34273    | 13284    | 3715     | 97     | 35   | 34   | 14   | 387    |
| MOT16-05 | 58.903 | 78.458 | 59.724 | 69.317   | 87.844   | 28.8   | 56.8   | 14.4   | 43.971  | 4726     | 2092     | 654      | 56     | 36   | 71   | 18   | 159    |
| MOT16-09 | 62.317 | 84.061 | 63.116 | 75.594   | 85.832   | 52     | 44     | 4      | 50.268  | 3974     | 1283     | 656      | 42     | 13   | 11   | 1    | 88     |
| MOT16-10 | 54.522 | 76.896 | 55.041 | 59.336   | 93.251   | 25.926 | 50     | 24.074 | 40.813  | 7309     | 5009     | 529      | 64     | 14   | 27   | 13   | 339    |
| MOT16-11 | 66.514 | 85.207 | 66.819 | 78.363   | 87.161   | 49.275 | 34.783 | 15.942 | 54.922  | 7189     | 1985     | 1059     | 28     | 34   | 24   | 11   | 77     |
| MOT16-13 | 40.672 | 75.058 | 41.162 | 45.721   | 90.933   | 18.692 | 44.86  | 36.449 | 29.269  | 5235     | 6215     | 522      | 56     | 20   | 48   | 39   | 204    |
| COMBINED | 55.497 | 78.022 | 55.892 | 62.749   | 90.15    | 31.141 | 45.648 | 23.211 | 41.706  | 69279    | 41128    | 7570     | 436    | 161  | 236  | 120  | 1432   |

IDF1: **59.31**



## License

[MIT License](./LICENSE)


## Citation

If you find my work useful in your research, please consider citing:
```
@misc{SuperbTUM,
  author = {Mingzhe Hu},
  title = {real-time ReID Tracking},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SuperbTUM/real-time-ReID-tracking}},
}
```
