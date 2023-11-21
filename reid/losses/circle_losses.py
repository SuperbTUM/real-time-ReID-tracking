import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple


class CircleLoss(nn.Module):
    def __init__(self, m: float = 0.35, gamma: float = 64) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    @staticmethod
    def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        similarity_matrix = normed_feature @ normed_feature.T
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=0)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def get_logits(self, normed_feature, label):
        sp, sn = self.convert_label_to_similarity(normed_feature, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        label = F.one_hot(label, num_classes=normed_feature.size(1))
        pred_class_logits = label * logit_p + (1.0 - label) * logit_n # incorrect
        return pred_class_logits

    def forward_focalize(self, normed_feature, label):
        class_logits = self.get_logits(normed_feature, label)
        ce_loss = F.cross_entropy(class_logits, label, reduction="none", label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

    def forward(self, normed_feature: Tensor, label: Tensor) -> Tensor:
        sp, sn = self.convert_label_to_similarity(normed_feature, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss / label.size(0)
