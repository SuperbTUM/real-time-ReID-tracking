import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx

from .utils import softmax_weights, euclidean_dist


cudnn.deterministic = True
cudnn.benchmark = True


class WeightedRegularizedTripletXBM(object):

    def __init__(self, reduction="mean"):
        self.ranking_loss = nn.SoftMarginLoss(reduction=reduction)

    def __call__(self, global_feat, labels, global_feat_row, labels_row, normalize_feature=False, weights=None):
        if normalize_feature:
            global_feat = F.normalize(global_feat, dim=-1)
            global_feat_row = F.normalize(global_feat_row, dim=-1)
        dist_mat = euclidean_dist(global_feat, global_feat_row)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = (labels.expand(labels_row.size(0), N).t()).eq(labels_row.expand(N, labels_row.size(0))).float()
        is_neg = 1 - is_pos
        is_pos[:, :N] = is_pos[:, :N] - torch.eye(N, dtype=torch.uint8).cuda()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)
        if weights is not None:
            loss *= weights
            return loss.sum()
        return loss
