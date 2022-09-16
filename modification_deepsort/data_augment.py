import os.path as osp
import glob
import re
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class BaseDataset:
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset


class Augmentation(Dataset):
    def __init__(self, raw_dataset, root=None, transform=None, smooth=None):
        super(Augmentation, self).__init__()
        self.raw_dataset = raw_dataset
        self.cam_pid = defaultdict(set)
        self.campid_index = defaultdict(dict)
        for index, (path, pid, camid) in enumerate(raw_dataset):
            self.cam_pid[camid].add(pid)
            if pid not in self.campid_index[camid]:
                self.campid_index[camid][pid] = set()
            self.campid_index[camid][pid].add(index)
        self.root = root
        self.transform = transform
        self.smooth = smooth

    def augment(self, index):
        path, pid, camid = self.raw_dataset[index]
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
        start_point = (ref_h - resized_upper_body_img.shape[0], random.randint(0, ref_w - resized_upper_body_img.shape[1]))
        referenced_image[start_point[0]:, start_point[1]:start_point[1]+resized_upper_body_img.shape[1],:] = resized_upper_body_img
        if self.smooth:
            referenced_image = self.smooth_stitching(start_point[0],
                                                     ref_h,
                                                     start_point[1],
                                                     start_point[1]+resized_upper_body_img.shape[1],
                                                     referenced_image)
        return Image.fromarray(referenced_image)

    @staticmethod
    def smooth_stitching(start_h,
                         end_h,
                         start_w,
                         end_w,
                         referenced_image):
        ref_h, ref_w, _ = referenced_image.shape
        for index_h, h in enumerate(range(start_h, end_h)):
            if referenced_image.dtype == int:
                referenced_image[h, start_w] = int(referenced_image[h-1:min(ref_h, h+2), max(0, start_w-1):start_w+2].mean())
                referenced_image[h, end_w - 1] = int(referenced_image[h-1:min(ref_h, h+2), end_w-2:min(ref_w, end_w+1)].mean())
            else:
                referenced_image[h, start_w] = referenced_image[h-1:min(ref_h, h+2), max(0, start_w-1):start_w+2].mean()
                referenced_image[h, end_w - 1] = referenced_image[h-1:min(ref_h, h+2), end_w-2:min(ref_w, end_w+1)].mean()
        for index_w, w in enumerate(range(start_w, end_w)):
            if referenced_image.dtype == int:
                referenced_image[start_h, w] = int(referenced_image[start_h-1:start_h+2, max(0, w-1):min(ref_w, w+2)].mean())
            else:
                referenced_image[start_h, w] = referenced_image[start_h-1:start_h+2, max(0, w-1):min(ref_w, w+2)].mean()
        return referenced_image

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        path, pid, camid = self.raw_dataset[index]
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
    start_point = (ref_h - resized_upper_body_img.shape[0], random.randint(0, ref_w - resized_upper_body_img.shape[1]))
    referenced_image[start_point[0]:, start_point[1]:start_point[1] + resized_upper_body_img.shape[1],
    :] = resized_upper_body_img
    start_h, end_h = start_point[0], ref_h
    start_w, end_w = start_point[1], start_point[1] + resized_upper_body_img.shape[1]
    ref_h, ref_w, _ = referenced_image.shape
    for index_h, h in enumerate(range(start_h, end_h)):
        if referenced_image.dtype == int:
            referenced_image[h, start_w] = int(referenced_image[h-1:min(ref_h, h+2), max(0, start_w-1):start_w+2].mean())
            referenced_image[h, end_w - 1] = int(referenced_image[h-1:min(ref_h, h+2), end_w-2:min(ref_w, end_w+1)].mean())
        else:
            referenced_image[h, start_w] = referenced_image[h-1:min(ref_h, h+2), max(0, start_w-1):start_w+2].mean()
            referenced_image[h, end_w - 1] = referenced_image[h-1:min(ref_h, h+2), end_w-2:min(ref_w, end_w+1)].mean()
    for index_w, w in enumerate(range(start_w, end_w)):
        if referenced_image.dtype == int:
            referenced_image[start_h, w] = int(referenced_image[start_h-1:start_h+2, max(0, w-1):min(ref_w, w+2)].mean())
        else:
            referenced_image[start_h, w] = referenced_image[start_h-1:start_h+2, max(0, w-1):min(ref_w, w+2)].mean()
    return Image.fromarray(referenced_image)


if __name__ == "__main__":
    augmented = demo("../demo/camera1_class0_frame000_person003.jpg", "../demo/camera1_class0_frame001_person002.jpg")
    augmented.show()
