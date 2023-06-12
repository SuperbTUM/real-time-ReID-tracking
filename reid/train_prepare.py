import torch
from torch import Tensor
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from prefetch_generator import BackgroundGenerator
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.onnx
import numpy as np
import random
import math
from PIL import Image
from bisect import bisect_right
from typing import Tuple
from collections import defaultdict

cudnn.deterministic = True
cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class FocalLoss(nn.Module):
    def __init__(self, smoothing=0.1, epsilon=0., alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, self.alpha, reduction='none', label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        poly_loss = focal_loss + self.epsilon * torch.pow(1-pt, self.gamma+1)
        return poly_loss.mean()


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.1, epsilon=0., k_sparse=-1):
        """ Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.k_sparse = k_sparse

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        target = target.long()
        if self.k_sparse > 0:
            topk = x.topk(self.k_sparse, dim=-1)[0]
            pos_loss = torch.logsumexp(topk, dim=-1)
            neg_loss = torch.gather(x, 1, target[:, None].expand(-1, x.size(1)))[:, 0]
            nll_loss = (pos_loss - neg_loss).sum()
        else:
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        smoothed_labels = F.one_hot(target, x.size(-1)) * self.confidence + self.smoothing / x.size(-1)
        one_minus_pt = torch.sum(smoothed_labels * (1 - logprobs), dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        poly_loss = loss + one_minus_pt * self.epsilon
        # return loss.mean()
        return poly_loss.mean()


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device="cuda"))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class WeightedRegularizedTriplet(object):

    def __init__(self):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = F.normalize(global_feat, dim=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

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

        return loss


def cosine_dist(x, y):
    bs1, bs2 = x.shape[0], y.shape[0]
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return (1 - cosine) / 2


class TripletLossPenalty(nn.Module):
    def __init__(self, beta, margin=0.3, reduction="mean"):
        super(TripletLossPenalty, self).__init__()
        self.beta = beta
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1, x2, y):
        """
        :param x1: negative
        :param x2: positive
        :param y:
        :return:
        """
        penalized_margin = (1-self.beta)*self.margin / (1+self.beta)
        loss = torch.maximum(torch.zeros_like(y), -y * ((1-self.beta)*x1 - (1+self.beta)*x2) + penalized_margin)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, alpha=0.4, smooth=False, sigma=1.0, reduction="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if alpha == 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.ranking_loss = TripletLossPenalty(alpha, margin, reduction)
        self.alpha = alpha
        self.smooth = smooth
        self.sigma = sigma

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # dist = 0.9 * dist + 0.1 * cosine_dist(inputs, inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        if self.sigma < 1.:
            loss = self.ranking_loss(torch.exp(dist_an / self.sigma), torch.exp(dist_ap / self.sigma), y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        # loss += self.alpha * torch.mean(dist_an + dist_ap) / 2
        if self.smooth:
            loss = F.softplus(loss)
        else:
            loss = F.relu(loss)
        return loss


class TripletBeta(TripletLoss):
    def __init__(self, margin=0.3, alpha=0.4, smooth=False, sigma=1.0, reduction="mean"):
        super(TripletBeta, self).__init__(margin, alpha, smooth, sigma, reduction)

    def forward(self, inputs, targets, inputs_augment):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
            inputs_augment:
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        dist_augment = (inputs * inputs_augment).sum(dim=1, keepdim=True).expand(n, n)
        dist_augment = dist_augment + dist_augment.t()
        dist_augment.addmm_(inputs, inputs_augment.t(), beta=1, alpha=-2)
        dist_augment = dist_augment.clamp(min=1e-12).sqrt()

        # dist = 0.9 * dist + 0.1 * cosine_dist(inputs, inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            positive = dist[i][mask[i]].max()
            if positive < 1e-6:
                positive = dist_augment[i][mask[i]].max()
            dist_ap.append(positive)
            dist_an.append(max(dist[i][mask[i] == 0].min(), dist_augment[i][mask[i] == 0].min()))
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        if self.sigma < 1.:
            loss = self.ranking_loss(torch.exp(dist_an / self.sigma), torch.exp(dist_ap / self.sigma), y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        # loss += self.alpha * torch.mean(dist_an + dist_ap) / 2
        if self.smooth:
            loss = F.softplus(loss)
        else:
            loss = F.relu(loss)
        return loss


class SemiHardTriplet(nn.Module):
    def __init__(self, margin, device="cuda"):
        super(SemiHardTriplet, self).__init__()
        self.margin = margin
        self.device = device

    def pairwise_distance_torch(self, embeddings):
        """Computes the pairwise distance matrix with numerical stability.
        output[i, j] = || feature[i, :] - feature[j, :] ||_2
        Args:
          embeddings: 2-D Tensor of size [number of data, feature dimension].
        Returns:
          pairwise_distances: 2-D Tensor of size [number of data, number of data].
        """

        # pairwise distance matrix with precise embeddings
        precise_embeddings = embeddings.to(dtype=torch.float32)

        c1 = torch.pow(precise_embeddings, 2).sum(dim=-1)
        c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(dim=0)
        c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

        c1 = c1.reshape((c1.shape[0], 1))
        c2 = c2.reshape((1, c2.shape[0]))
        c12 = c1 + c2
        pairwise_distances_squared = c12 - 2.0 * c3

        # Deal with numerical inaccuracies. Set small negatives to zero.
        pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(self.device))
        # Get the mask where the zero distances are at.
        error_mask = pairwise_distances_squared.clone()
        error_mask[error_mask > 0.0] = 1.
        error_mask[error_mask <= 0.0] = 0.

        pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

        # Explicitly set diagonals to zero.
        mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(
            torch.ones(pairwise_distances.shape[0]))
        pairwise_distances = torch.mul(pairwise_distances.to(self.device), mask_offdiagonals.to(self.device))
        return pairwise_distances

    def forward(self, embeddings, targets):
        """Computes the triplet loss_functions with semi-hard negative mining.
           The loss_functions encourages the positive distances (between a pair of embeddings
           with the same labels) to be smaller than the minimum negative distance
           among which are at least greater than the positive distance plus the
           margin constant (called semi-hard negative) in the mini-batch.
           If no such negative exists, uses the largest negative distance instead.
           See: https://arxiv.org/abs/1503.03832.
           We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
           [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
           2-D float `Tensor` of l2 normalized embedding vectors.
           Args:
             margin: Float, margin term in the loss_functions definition. Default value is 1.0.
             name: Optional name for the op.
           """

        # Reshape label tensor to [batch_size, 1].
        lshape = targets.shape
        labels = torch.reshape(targets, [lshape[0], 1])

        pdist_matrix = self.pairwise_distance_torch(embeddings)

        # Build pairwise binary adjacency matrix.
        adjacency = torch.eq(labels, labels.transpose(0, 1))
        # Invert so we can select negatives only.
        adjacency_not = adjacency.logical_not()

        batch_size = labels.shape[0]

        # Compute the mask.
        pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
        adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

        transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
        greater = pdist_matrix_tile > transpose_reshape

        mask = adjacency_not_tile & greater

        # final mask
        mask_step = mask.to(dtype=torch.float32)
        mask_step = mask_step.sum(axis=1)
        mask_step = mask_step > 0.0
        mask_final = mask_step.reshape(batch_size, batch_size)
        mask_final = mask_final.transpose(0, 1)

        adjacency_not = adjacency_not.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
        masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + \
                          axis_maximums[0]
        negatives_outside = masked_minimums.reshape([batch_size, batch_size])
        negatives_outside = negatives_outside.transpose(0, 1)

        # negatives_inside: largest D_an.
        axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
        masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + \
                          axis_minimums[0]
        negatives_inside = masked_maximums.repeat(1, batch_size)

        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = self.margin + pdist_matrix - semi_hard_negatives

        mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(self.device)
        num_positives = mask_positives.sum() + 1e-6

        triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives),
                                  torch.tensor([0.]).to(self.device))).sum() / num_positives
        triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
        return triplet_loss


class HybridLoss(nn.Module):
    def __init__(self, num_classes,
                 feat_dim=512,
                 margin=0.3,
                 smoothing=0.1,
                 epsilon=0,
                 lamda=0.0005,
                 alpha=0.4,
                 triplet_smooth=False,
                 class_stats=None):
        super().__init__()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        if margin > 0.:
            # self.triplet = TripletLoss(margin, alpha, triplet_smooth)  # Smooth only works for hard triplet loss now
            self.triplet = TripletBeta(margin, alpha, triplet_smooth)
        else:
            self.triplet = WeightedRegularizedTriplet()
        self.smooth = FocalLoss(smoothing, epsilon, class_stats)#LabelSmoothing(smoothing, epsilon)
        self.lamda = lamda

    def forward(self, embeddings, outputs, targets, embeddings_augment=None):
        """
        features: feature vectors
        targets: ground truth labels
        """
        smooth_loss = self.smooth(outputs, targets)
        # triplet_loss = self.triplet(embeddings, targets)
        triplet_loss = self.triplet(embeddings, targets, embeddings_augment)
        center_loss = self.center(embeddings, targets)
        return smooth_loss + triplet_loss + self.lamda * center_loss


class RepreLoss(nn.Module):
    def __init__(self, lamda, margin=0.3, feat_dim=512):
        super(RepreLoss, self).__init__()
        self.triplet = TripletLoss(margin, alpha=0.0)
        self.center = CenterLoss(feat_dim=feat_dim)
        self.lamda = lamda

    def forward(self, embeddings, targets):
        return self.triplet(embeddings, targets) + self.lamda * self.center(embeddings, targets)


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    @staticmethod
    def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def forward(self, normed_feature: Tensor, label: Tensor) -> Tensor:

        sp, sn = self.convert_label_to_similarity(normed_feature, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


def to_onnx(model, input_dummy, input_names=["input"], output_names=["outputs"]):
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
                          "checkpoint/reid_model.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes
                          )
    except:
        torch.onnx.export(model,
                          input_dummy,
                          "checkpoint/reid_model.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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
        new = img.convert("L")   # Convert from here to the corresponding grayscale image
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
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
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
