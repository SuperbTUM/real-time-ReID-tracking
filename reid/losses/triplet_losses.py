import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.onnx

from .utils import softmax_weights, euclidean_dist


cudnn.deterministic = True
cudnn.benchmark = True


class WeightedRegularizedTriplet(object):

    def __init__(self, reduction="mean"):
        self.ranking_loss = nn.SoftMarginLoss(reduction=reduction)

    def __call__(self, global_feat, labels, normalize_feature=False, weights=None):
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
        if weights is not None:
            loss *= weights
            return loss.sum()
        return loss


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
        penalized_margin = (1 - self.beta) * self.margin / (1 + self.beta)
        loss = torch.maximum(torch.zeros_like(y), -y * ((1 - self.beta) * x1 - (1 + self.beta) * x2) + penalized_margin)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, alpha=0., smooth=False, sigma=1.0, reduction="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if alpha == 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.ranking_loss = TripletLossPenalty(alpha, margin, reduction)
        self.alpha = alpha
        self.smooth = smooth
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, global_feat, labels, weights=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        if self.reduction == "none":
            loss = loss * weights
            loss = loss.sum()
        # if self.smooth:
        #     loss = F.softplus(loss)
        # else:
        #     loss = F.relu(loss)
        return loss


class TripletBeta(TripletLoss):
    def __init__(self, margin=0.3, alpha=0.4, smooth=False, sigma=1.0, reduction="mean"):
        super(TripletBeta, self).__init__(margin, alpha, smooth, sigma, reduction)

    def forward(self, inputs, targets, inputs_augment=None, weights=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
            inputs_augment: Optional
            weights: Optional
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        if inputs_augment is not None:
            dist_augment = (inputs * inputs_augment).sum(dim=1, keepdim=True).expand(n, n)
            dist_augment = dist_augment + dist_augment.t()
            dist_augment.addmm_(inputs, inputs_augment.t(), beta=1, alpha=-2)
            dist_augment = dist_augment.clamp(min=1e-12).sqrt()
        else:
            dist_augment = None

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            positive = dist[i][mask[i]].max()
            if dist_augment is not None:
                positive = max(positive, dist_augment[i][mask[i]].max())
            dist_ap.append(positive)
            if dist_augment is not None:
                dist_an.append(max(dist[i][mask[i] == 0].min(), dist_augment[i][mask[i] == 0].min()))
            else:
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
        if weights is not None:
            loss *= weights
            loss = loss.sum()
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
