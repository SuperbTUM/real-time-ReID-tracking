# Real-time ReID Tracking w/. Lite but Strong Feature Extractor & GAN
[Slides](https://docs.google.com/presentation/d/1cs5wEIcJWV5H6F5xnWDV0ikYf-uWuFvZj3byTBs4sMQ/edit#slide=id.p), Manuscript(Coming Soon...)

## Introduction

From Sort to OCSort, we are becoming aware that deep feature extractor is crucial in both re-identification and multiple object tracking.
The project integrates Yolo detection, GAN, deep feature extractor for re-identification and MOT. 
The baseline is a Yolov5(now is YoloV8!) based DeepSort(now is DeepOCSort & StrongSort) algorithm.
Everything can be found [here](https://github.com/mikel-brostrom/yolo_tracking). 
The author has updated the repository with import of fast-reid package.  
**We yield even better results than ResNet50 with ResNet18-based model!!!**



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
Now we primarily use MOT16 evaluation as benchmark and Market1501/DukeMTMC/VeRi-776 to pre-train our re-id network.

Video Dataset: [MOT16](https://motchallenge.net/data/MOT16/)

Image Datasets: [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ), [DukeMTMC-reID](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view),
[VeRi-776](https://vehiclereid.github.io/VeRi/)



## Quick Start

[External Quick Start Doc](./QUICK_START.md)


## Evaluation

**Re-identification**

[External ReID Doc](./REID_EVAL.md)

**Tracking**

[External Tracking Doc](./TRACKING_EVAL.md)


## Tracking Speed

[The following conclusion is outdated]

The baseline extractor in DeepSort-YoloV5 implementation is pure ResNet-18. 
The inference speed is 15 ~ 20 ms per frame [need to be re-assessed and OSNet 1.0 is heavy!!! ~100ms per frame] depending on the sparsity of pedestrians with 640 * 640 resolution with Tesla P100. 
It may be slower if bounding box is resized to (128, 256). 
The modified extractor is based on Dense skip connection in ResNet-18 with Squeeze and Excitation Network, only a minor increase on the number of learnable parameters. 
The tracking speed is 17 ms per frame under the same testing environment. 
The speed is acquired with `time` package after the synchronization of CUDA.


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
