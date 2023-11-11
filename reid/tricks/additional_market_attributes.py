import os
import sys
sys.path.insert(0, "/".join((os.getcwd(), "../losses")))

import torch
import torch.nn.functional as F
from scipy import io
from losses.utils import euclidean_dist


def get_attributes(file_name):
    mat = io.loadmat(file_name)["market_attribute"][0][0]
    mat = mat[0][0][0]
    identity_list = list(map(lambda x: int(x.item()), mat[-1][0]))
    attributes = []
    age_mapping = {1: [0, 0, 0, 1], 2: [0, 0, 1, 0], 3: [0, 1, 0, 0], 4: [1, 0, 0, 0]}
    for klass in range(len(mat[0][0])):
        presentation = []
        for i in range(27):
            if i == 0:
                presentation.extend(age_mapping[mat[i][0][klass]])
            else:
                presentation.append(mat[i][0][klass])
        attributes.append(torch.tensor(presentation))
    attributes = torch.stack(attributes)
    return {identity: template for identity, template in zip(identity_list, attributes)}


def get_attribute_dist(labels, file_name):
    # labels are only used for maintaining correct orders
    attributes_dict = get_attributes(file_name)
    reordered_attributes = []
    for label in labels:
        reordered_attributes.append(attributes_dict.get(label, torch.ones(30)))
    reordered_attributes = torch.stack(reordered_attributes)
    reordered_attributes = F.normalize(reordered_attributes, dim=1)
    attributes_dist = euclidean_dist(reordered_attributes, reordered_attributes)
    return attributes_dist.numpy()
