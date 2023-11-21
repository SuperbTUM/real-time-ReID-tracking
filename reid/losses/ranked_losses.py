import torch
from .utils import normalize_rank, euclidean_dist


def rank_loss(dist_mat, labels, margin, alpha, tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]

    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        # is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])

        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        an_is_pos = torch.lt(dist_an, alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        loss_an = torch.div(an_ln_sum, an_weight_sum)

        total_loss = total_loss + loss_ap + loss_an
    total_loss = total_loss * 1.0 / N
    return total_loss


class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, margin=1.3, alpha=2.0, tval=1.0):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval

    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        total_loss = rank_loss(dist_mat, labels, self.margin, self.alpha, self.tval)

        return total_loss
