import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from train_utils import redetection, recrop


class reidDataset(Dataset):
    def __init__(self, images, train_classes, transform=None, get_crop=False):
        self.images = images
        self.train_classes = train_classes
        self.transform = transform
        self.images_pseudo = []
        self._continual = False
        self.cropped = []
        self.cropped_pseudo = []
        self.class_stats = [0 for _ in range(train_classes)]
        for image in images:
            if image[1] < train_classes:
                self.class_stats[image[1]] += 1
        self.get_crop = get_crop
        if get_crop:
            pure_images = list(map(lambda x: x[0], images))
            i = 0
            while i < len(pure_images):
                local_batch = []
                end = min(i+64, len(pure_images))
                for j in range(i, end):
                    local_batch.append(Image.open(pure_images[j]).convert("RGB"))
                cropped_imgs = redetection(local_batch, "pil")
                self.cropped.extend(cropped_imgs)
                i = end

    def get_class_stats(self):
        return self.class_stats

    def set_cross_domain(self):
        self._continual = True

    def reset_cross_domain(self):
        self._continual = False

    def __len__(self):
        if self._continual:
            return len(self.images_pseudo) + len(self.images)
        return len(self.images)

    def add_pseudo(self, pseudo_labeled_data, num_class_new):
        self.images_pseudo.extend(pseudo_labeled_data)
        self.class_stats = self.class_stats + [0 for _ in range(num_class_new - self.train_classes)]
        for image in self.images_pseudo:
            if image[1] >= self.train_classes:
                self.class_stats[image[1]] += 1
        if self.get_crop:
            pure_images = list(map(lambda x: x[0], self.images_pseudo))
            i = 0
            while i < len(pure_images):
                local_batch = []
                end = min(i + 64, len(pure_images))
                for j in range(i, end):
                    local_batch.append(Image.open(pure_images[j]).convert("RGB"))
                cropped_imgs = redetection(local_batch, "pil")
                self.cropped_pseudo.extend(cropped_imgs)
                i = end

    def __getitem__(self, item):
        if self._continual:
            if item < len(self.images):
                detailed_info = list(self.images[item])
            else:
                detailed_info = list(self.images_pseudo[item - len(self.images)])
        else:
            detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        # if self.get_crop and np.random.random() > 0.5:
        #     if item < len(self.images):
        #         detailed_info[0] = self.cropped[item]
        #     else:
        #         detailed_info[0] = self.cropped_pseudo[item - len(self.images)]
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        for i in range(2, len(detailed_info)):
            detailed_info[i] = torch.tensor(detailed_info[i], dtype=torch.long)
        if self._continual:
            return detailed_info + [torch.tensor(0.) if item < len(self.images) else torch.tensor(1.)] # tricky
        return detailed_info


# @credit to Alibaba
def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    # This one is not performing well
    def __init__(self, data_source, num_instances=16):
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.pid_index = defaultdict(int)
        for index, data_info in enumerate(data_source):
            pid = data_info[1].item()
            self.index_dic[pid].append(index)
            self.pid_index[index] = pid
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.index_dic[self.pids[kid]])

            ret.append(i)

            pid_i = self.pid_index[i]
            index = self.index_dic[pid_i]

            select_indexes = No_index(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)

import copy
class RandomIdentitySampler_(torch.utils.data.sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, img_info in enumerate(self.data_source):
            pid = img_info[1]
            self.index_dic[pid.item()].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
