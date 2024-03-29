import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """In exploration of how to combine bnneck with focal loss"""
    def __init__(self, smoothing=0.1, epsilon=0., alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        one_hot_key = F.one_hot(targets, num_classes=num_classes)
        pt = one_hot_key * F.softmax(outputs, dim=-1)
        difficulty_level = torch.pow(1 - pt, self.gamma)
        lb_pos, lb_neg = 1. - self.smoothing, self.smoothing / (num_classes - 1)
        lb_one_hot = torch.empty_like(outputs).fill_(lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos).detach()
        logs = F.log_softmax(outputs, dim=-1)
        focal_loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        # mean over the batch
        focal_loss = focal_loss * self.alpha[targets]
        if self.alpha is not None and self.epsilon != 0.:
            # epsilon2 = 0.2 in polyloss is still tricky
            poly_loss = focal_loss + (
                        self.epsilon * torch.pow(1 - pt, self.gamma + 1) + 0.2 * torch.pow(1 - pt, self.gamma + 2)) * \
                        self.alpha[targets]
        elif self.epsilon != 0.:
            poly_loss = focal_loss + self.epsilon * torch.pow(1 - pt, self.gamma + 1) + 0.2 * torch.pow(1 - pt,
                                                                                                        self.gamma + 2)
        else:
            poly_loss = focal_loss
        return poly_loss.mean()


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        smoothing (float): weight.
    """
    def __init__(self, num_classes, smoothing=0.1, epsilon=0.0, use_gpu=True, tao=1.):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.tao = tao
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        probs = F.softmax(inputs/self.tao, dim=1)
        log_probs = self.logsoftmax(inputs/self.tao)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.smoothing) * targets + self.smoothing / self.num_classes
        loss = (- targets * log_probs).sum(1)
        one_minus_pt = torch.sum(targets * (1 - probs), dim=1)
        loss += one_minus_pt * self.epsilon
        if self.epsilon < 0:
            loss += 0.2 * one_minus_pt ** 2
        return loss.mean()


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.1, epsilon=0., class_weights=None):
        """ Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)
        target = target.long()

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        smoothed_labels = F.one_hot(target, x.size(-1)) * self.confidence + self.smoothing / x.size(-1)
        one_minus_pt = torch.sum(smoothed_labels * (1 - probs), dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        poly_loss = loss + one_minus_pt * self.epsilon
        if self.class_weights is not None:
            poly_loss *= self.class_weights[target]
        # return loss.mean()
        return poly_loss.mean()

