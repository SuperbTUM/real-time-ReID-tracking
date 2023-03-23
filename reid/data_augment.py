import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import defaultdict
from dataset_market import *


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


if __name__ == "__main__":
    augmented = demo("../demo/camera1_class0_frame000_person003.jpg", "../demo/camera1_class0_frame001_person002.jpg")
    augmented.show()
