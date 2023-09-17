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

**ReID(Image Retrival) training**

Although some checkpoints are available, you are still advised to train your Re-ID model with Market1501.
Due to privacy issue, some datasets such as DukeMTMC are no longer open to the public and not acceptable to the academy as well.
In the [reid](https://github.com/SuperbTUM/real-time-ReID-tracking/tree/main/reid) folder, you can see how we build the model as well train the model. 
There are a few versions of models.

The main focus of the project is to construct a lite backbone for mobile development and real-time tracking. 
But still, we include a model zoo, with CNN-based re-id models, and vision transformer based models, where you can access all of them in the [backbones](https://github.com/SuperbTUM/real-time-person-ReID-tracking/tree/main/reid/backbones) folder.
We train the model on both image-based dataset and video-based dataset(w/. ground truth), and the scripts can be access under the same folder.

For non-continual image training
```python
python reid/image_reid_train.py --bs 32 --backbone cares18 --accelerate --center_lamda 0.0005 --instance 8 --dataset market1501
```

For continual image training
```python
python reid/image_reid_train.py --bs 32 --backbone cares18 --accelerate --center_lamda 0.0005 --instance 8 --continual --eps 0.5 --dataset market1501
```

For image training with SIE
```python
python reid/image_reid_train.py --bs 32 --backbone seres18 --accelerate --center_lamda 0.0005 --instance 8 --continual --eps 0.5 --dataset market1501 --sie
```

For image-level testing 
```python
python reid/image_reid_inference.py --backbone cares18 --bs 16 --ckpt checkpoint/reid_model_xxx.onnx --eps 0.5 --dataset xxx
```

For video-level training
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
