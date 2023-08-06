import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import defaultdict
import os.path as osp


# I am not sure if this is reliable (for occlusion)
class Augmentation(Dataset):
    def __init__(self, raw_dataset, root=None, transform=None, foreground=False):
        super(Augmentation, self).__init__()
        self.raw_dataset = raw_dataset
        self.cam_pid = defaultdict(set)
        self.campid_index = defaultdict(dict)
        for index, (path, pid, camid, seqid) in enumerate(raw_dataset):
            self.cam_pid[camid].add(pid)
            if pid not in self.campid_index[camid]:
                self.campid_index[camid][pid] = set()
            self.campid_index[camid][pid].add(index)
        self.root = root
        self.transform = transform
        self.foreground = foreground

    def augment(self, index):
        path, pid, camid, seqid = self.raw_dataset[index]
        pids = self.cam_pid[camid]
        pids.discard(pid)
        available_pids = pids.copy()
        indices = []
        for avail_pid in available_pids:
            indices.extend(list(self.campid_index[camid][avail_pid]))
        index_augment = random.choice(indices)
        del indices
        if self.root:
            fpath = osp.join(self.root, self.raw_dataset[index_augment][0])
        else:
            fpath = self.raw_dataset[index_augment][0]
        helper_image = np.array(Image.open(fpath).convert("RGB"))
        del fpath
        h, w, _ = helper_image.shape
        upper_body = helper_image[:int(0.25*h), :, :]
        upper_body_pil = Image.fromarray(upper_body)
        if self.root:
            fpath = osp.join(self.root, self.raw_dataset[index][0])
        else:
            fpath = self.raw_dataset[index][0]
        referenced_image = np.array(Image.open(fpath).convert("RGB"))
        del fpath
        ref_h, ref_w, _ = referenced_image.shape
        if 0.25 * ref_h / upper_body.shape[0] < ref_w / upper_body.shape[1]:
            ratio = random.randint(int(0.25 * ref_h) >> 1, int(0.25 * ref_h)) / upper_body.shape[0]
        else:
            ratio = random.randint(int(ref_w) >> 1, int(ref_w)) / upper_body.shape[1]
        resized_upper_body = upper_body_pil.resize((int(upper_body.shape[1] * ratio), int(upper_body.shape[0] * ratio)))
        resized_upper_body_img = np.array(resized_upper_body)
        if random.random() > 0.5:
            ori_h, ori_w, ori_c = resized_upper_body_img.shape
            resized_upper_body_img_flip = resized_upper_body_img.copy()
            resized_upper_body_img_flip = resized_upper_body_img_flip.reshape((-1, 3))[::-1]
            resized_upper_body_img = resized_upper_body_img_flip.reshape(ori_h, ori_w, ori_c)[::-1]
        start_point = (ref_h - resized_upper_body_img.shape[0], random.randint(0, ref_w - resized_upper_body_img.shape[1]))
        if self.foreground:
            referenced_image = self.foreground_augment(resized_upper_body_img, start_point, referenced_image)
        else:
            referenced_image[start_point[0]:, start_point[1]:start_point[1]+resized_upper_body_img.shape[1],:] = resized_upper_body_img
        return Image.fromarray(referenced_image)

    @staticmethod
    def foreground_augment(resized_upper_body_image, start_point, referenced_image):
        h, w = resized_upper_body_image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (2, 2, w - 2, h - 2)
        fg_placeholder = np.zeros((1, 65), dtype="float")
        bg_placeholder = np.zeros((1, 65), dtype="float")
        cv2.grabCut(resized_upper_body_image, mask, rect, bg_placeholder, fg_placeholder, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        resized_upper_body_image = resized_upper_body_image * mask2[:, :, np.newaxis]
        for i, h in enumerate(range(start_point[0], referenced_image.shape[0])):
            for j, w in enumerate(range(start_point[1], start_point[1]+resized_upper_body_image.shape[1])):
                referenced_image[h, w, :] = resized_upper_body_image[i, j, :] if resized_upper_body_image[i, j, :].tolist() != [0, 0, 0] else referenced_image[h, w, :]
        return referenced_image

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        path, pid, camid, seqid = self.raw_dataset[index]
        if self.root:
            fpath = osp.join(self.root, path)
        else:
            fpath = path
        if random.random() > 0.5:
            image = self.augment(index)
        else:
            image = Image.open(fpath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, fpath, pid, camid, index


def demo(image_path1, image_path2):
    referenced_image = np.array(Image.open(image_path1).convert("RGB"))
    augment_image = np.array(Image.open(image_path2).convert("RGB"))
    h, w, _ = augment_image.shape
    upper_body = augment_image[:int(0.25 * h), :, :]
    upper_body_pil = Image.fromarray(upper_body)
    ref_h, ref_w, _ = referenced_image.shape
    if 0.25 * ref_h / upper_body.shape[0] < ref_w / upper_body.shape[1]:
        ratio = random.randint(int(0.25 * ref_h) >> 1, int(0.25 * ref_h)) / upper_body.shape[0]
    else:
        ratio = random.randint(int(ref_w) >> 1, int(ref_w)) / upper_body.shape[1]
    resized_upper_body = upper_body_pil.resize((int(upper_body.shape[1] * ratio), int(upper_body.shape[0] * ratio)))
    resized_upper_body_img = np.array(resized_upper_body)
    ori_h, ori_w, ori_c = resized_upper_body_img.shape
    resized_upper_body_img_flip = resized_upper_body_img.copy()
    resized_upper_body_img_flip = resized_upper_body_img_flip.reshape((-1, 3))[::-1]
    resized_upper_body_img = resized_upper_body_img_flip.reshape(ori_h, ori_w, ori_c)[::-1]
    start_point = (ref_h - resized_upper_body_img.shape[0], random.randint(0, ref_w - resized_upper_body_img.shape[1]))
    # referenced_image[start_point[0]:, start_point[1]:start_point[1] + resized_upper_body_img.shape[1],
    # :] = resized_upper_body_img

    h, w = resized_upper_body_img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (2, 2, w - 2, h - 2)
    fg_placeholder = np.zeros((1, 65), dtype="float")
    bg_placeholder = np.zeros((1, 65), dtype="float")
    cv2.grabCut(resized_upper_body_img, mask, rect, bg_placeholder, fg_placeholder, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    resized_upper_body_image = resized_upper_body_img * mask2[:, :, np.newaxis]
    for i, h in enumerate(range(start_point[0], referenced_image.shape[0])):
        for j, w in enumerate(range(start_point[1], start_point[1] + resized_upper_body_image.shape[1])):
            print(resized_upper_body_image[i, j, :])
            referenced_image[h, w, :] = resized_upper_body_image[i, j, :] \
                if resized_upper_body_image[i, j, :].tolist() != [0, 0, 0] else referenced_image[h, w, :]
    return Image.fromarray(referenced_image)


def foreground_segmentation(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (2, 2, w-2, h-2)
    fg_placeholder = np.zeros((1, 65), dtype="float")
    bg_placeholder = np.zeros((1, 65), dtype="float")
    cv2.grabCut(image, mask, rect, bg_placeholder, fg_placeholder, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    plt.imshow(image)
    plt.show()
    return image


import math
import torch
from torchvision import transforms

# This is the code of Local Grayscale Transformation

class LGT(object):

    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        :param img: should be a tensor for the sake of preprocessing convenience
        :return:
        """
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        new = img.convert("L")  # Convert from here to the corresponding grayscale image
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return transforms.ToTensor()(img)

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)
                img = np.asarray(img).astype('float')

                img[y1:y1 + h, x1:x1 + w, 0] = img_gray[y1:y1 + h, x1:x1 + w, 0]
                img[y1:y1 + h, x1:x1 + w, 1] = img_gray[y1:y1 + h, x1:x1 + w, 1]
                img[y1:y1 + h, x1:x1 + w, 2] = img_gray[y1:y1 + h, x1:x1 + w, 2]

                img = Image.fromarray(img.astype('uint8'))

                return transforms.ToTensor()(img)

        return transforms.ToTensor()(img)


def toSketch(img):  # Convert visible  image to sketch image
    img_np = np.asarray(img)
    img_inv = 255 - img_np
    img_blur = cv2.GaussianBlur(img_inv, ksize=(27, 27), sigmaX=0, sigmaY=0)
    img_blend = cv2.divide(img_np, 255 - img_blur, scale=256)
    img_blend = Image.fromarray(img_blend)
    return img_blend


"""
Randomly select several channels of visible image (R, G, B), gray image (gray), and sketch image (sketch) 
to fuse them into a new 3-channel image.
"""


def random_choose(r, g, b, gray_or_sketch):
    p = [r, g, b, gray_or_sketch, gray_or_sketch]
    idx = [0, 1, 2, 3, 4]
    random.shuffle(idx)
    return Image.merge('RGB', [p[idx[0]], p[idx[1]], p[idx[2]]])


# 10%(Grayscale) 5%(Grayscale-RGB) 5%(Sketch-RGB)
class Fuse_RGB_Gray_Sketch(object):
    def __init__(self, G=0.1, G_rgb=0.05, S_rgb=0.05):
        self.G = G
        self.G_rgb = G_rgb
        self.S_rgb = S_rgb

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        r, g, b = img.split()
        gray = img.convert('L')  # convert visible  image to grayscale images
        p = random.random()
        if p < self.G:  # just Grayscale
            img = Image.merge('RGB', [gray, gray, gray])

        elif p < self.G + self.G_rgb:  # fuse Grayscale-RGB
            img2 = random_choose(r, g, b, gray)
            img = img2

        elif p < self.G + self.G_rgb + self.S_rgb:  # fuse Sketch-RGB
            sketch = toSketch(gray)
            img3 = random_choose(r, g, b, sketch)
            img = img3
        return transforms.ToTensor()(img)


# 35%(Local Grayscale) 5%(Global Grayscale)
class Fuse_Gray(object):
    def __init__(self, LG=0.35, GG=0.05):
        self.LG = LG
        self.GG = GG
        self.lgt = LGT(1.)

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        gray = img.convert('L')  # convert visible  image to grayscale images
        p = random.random()
        if p < self.LG:  # Local Grayscale
            img = self.lgt(img)

        elif p < self.LG + self.GG:  # Global Grayscale
            img = Image.merge('RGB', [gray, gray, gray])

        if isinstance(img, torch.Tensor):
            return img
        return transforms.ToTensor()(img)
