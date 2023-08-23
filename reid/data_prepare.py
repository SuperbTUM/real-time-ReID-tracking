import numpy as np
import random
import torch
from collections import defaultdict


# @credit to Alibaba
def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
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
