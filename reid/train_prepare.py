import torch
import torch.backends.cudnn as cudnn

import onnx
from onnxsim import simplify
import random
import math
from bisect import bisect_right

cudnn.deterministic = True
cudnn.benchmark = True


def to_onnx(model, input_dummy, dataset_name, input_names=["input"], output_names=["outputs"]):
    import os
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    try:
        dynamic_axes = dict()
        for input_name in input_names:
            dynamic_axes[input_name] = {0: 'batch_size'}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: 'batch_size'}
        torch.onnx.export(model,
                          input_dummy,
                          "checkpoint/reid_model_{}.onnx".format(dataset_name),
                          export_params=True,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes
                          )
    except:
        torch.onnx.export(model,
                          input_dummy,
                          "checkpoint/reid_model_{}.onnx".format(dataset_name),
                          export_params=True,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names)
    # experimental
    model = onnx.load("checkpoint/reid_model_{}.onnx".format(dataset_name))
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "checkpoint/reid_model_{}_simp.onnx".format(dataset_name))


class WarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 gamma=0.1,
                 warmup_factor=0.01,
                 warmup_iters=10,
                 warmup_method="linear",
                 last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            for base_lr in self.base_lrs
        ]


class WarmUpCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 train_epochs,
                 delayed_epoch=30,
                 eta_min=7e-7,
                 gamma=0.1,
                 warmup_factor=0.01,
                 warmup_iters=10,
                 warmup_method="linear",
                 last_epoch=-1):
        self.last_epoch = last_epoch
        self.start_iters = max(warmup_iters, delayed_epoch)
        self.warmup = WarmUpScheduler(optimizer, gamma, warmup_factor, warmup_iters, warmup_method, last_epoch)
        self.cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs-self.start_iters, eta_min=eta_min)

    def step(self, epoch=None):
        self.last_epoch += 1
        if self.last_epoch < self.start_iters:
            self.warmup.step()
        else:
            self.cosine_sched.step()

    def get_lr(self):
        if self.last_epoch < self.start_iters:
            return self.warmup.get_lr()
        else:
            return self.cosine_sched.get_lr()

    def get_last_lr(self):
        if self.last_epoch < self.start_iters:
            return self.warmup.get_last_lr()
        else:
            return self.cosine_sched.get_last_lr()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
