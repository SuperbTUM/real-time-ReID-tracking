import torch
import torch.nn as nn

import torch.onnx


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True, ckpt=None, centroids=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.ckpt = ckpt

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device="cuda"))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if ckpt:
            self.load(centroids)

    def load(self, centroids=None):
        ckpt_centers = torch.load(self.ckpt)
        ckpt_classes = ckpt_centers.size(0)
        self.centers = nn.Parameter(torch.cat((ckpt_centers,
                                               torch.randn(self.num_classes - ckpt_classes, self.feat_dim,
                                                           device="cuda") if centroids is None else centroids.cuda()),
                                              dim=0))

    def save(self):
        torch.save(self.centers, "center_ckpt.pt")

    def forward(self, x, labels, weights=None):
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

        dist = distmat * mask.float()
        dist = dist.clamp(min=1e-12, max=1e+12)
        indices = torch.zeros(batch_size, dtype=torch.int32)
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            if value.detach().cpu().numpy():
                indices[i] = 1
        if weights is not None:
            dist = dist * weights[indices == 1]
            loss = dist.sum()
        else:
            loss = dist.sum() / batch_size
        return loss
