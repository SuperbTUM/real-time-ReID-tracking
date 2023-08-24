import torch.nn as nn

from .center_losses import CenterLoss
from .identification_losses import LabelSmoothing, LabelSmoothingMixup, FocalLoss, CrossEntropyLabelSmooth
from .triplet_losses import TripletBeta, WeightedRegularizedTriplet, TripletLoss
from .circle_losses import CircleLoss
from .utils import normalize_rank


class HybridLoss(nn.Module):
    def __init__(self, num_classes,
                 feat_dim=512,
                 margin=0.3,
                 smoothing=0.1,
                 epsilon=0,
                 lamda=0.0005,
                 alpha=0.0,
                 triplet_smooth=False,
                 class_stats=None,
                 circle_factor=0.):
        super().__init__()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        if margin > 0.:
            self.triplet = TripletLoss(margin, alpha, triplet_smooth)  # Smooth only works for hard triplet loss now
        else:
            self.triplet = WeightedRegularizedTriplet()
        self.smooth = CrossEntropyLabelSmooth(num_classes, smoothing, epsilon)#FocalLoss(smoothing, epsilon, class_stats)  # LabelSmoothing(smoothing, epsilon)
        self.circle = CircleLoss()
        self.lamda = lamda
        self.circle_factor = circle_factor

    def forward(self,
                embeddings,
                outputs,
                targets,
                normed_embeddings=None,
                embeddings_augment=None,
                outputs_augment=None):
        """
        features: feature vectors
        targets: ground truth labels
        """
        smooth_loss = self.smooth(outputs, targets)
        circle_loss = self.circle(normalize_rank(outputs, 1), targets)
        triplet_loss = self.triplet(embeddings, targets)
        center_loss = self.center(embeddings, targets)
        return (1. - self.circle_factor) * smooth_loss + \
               triplet_loss + \
               self.lamda * center_loss + \
               self.circle_factor * circle_loss


class HybridLossWeighted(nn.Module):
    def __init__(self, num_classes,
                 feat_dim=512,
                 margin=0.3,
                 smoothing=0.1,
                 epsilon=0,
                 lamda=0.0005,
                 alpha=0.0,
                 triplet_smooth=False,
                 mixup=False,
                 class_stats=None,
                 circle_factor=0.):
        super().__init__()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, ckpt="center_ckpt.pt")
        if margin > 0.:
            # self.triplet = TripletLoss(margin, alpha, triplet_smooth)  # Smooth only works for hard triplet loss now
            self.triplet = TripletBeta(margin, alpha, triplet_smooth, reduction="none")
        else:
            self.triplet = WeightedRegularizedTriplet()
        if mixup:
            self.smooth = LabelSmoothingMixup(smoothing, epsilon)
        else:
            self.smooth = LabelSmoothing(smoothing, epsilon) # FocalLoss(smoothing, epsilon, class_stats)  #
        self.circle = CircleLoss()
        self.lamda = lamda
        self.mixup = mixup
        self.circle_factor = circle_factor

    def forward(self,
                embeddings,
                outputs,
                targets,
                normed_embeddings=None,
                embeddings_augment=None,
                weights=None,
                outputs_augment=None,
                targets_a=None,
                targets_b=None,
                lam=None):
        """
        features: feature vectors
        targets: ground truth labels
        """
        if self.mixup:
            smooth_loss = self.smooth(outputs, targets_a, targets_b, lam)
        else:
            smooth_loss = self.smooth(outputs, targets)
        if outputs_augment is not None:
            circle_loss = self.circle(normalize_rank(outputs, 1), targets, normalize_rank(outputs_augment, 1))
        else:
            circle_loss = self.circle(normalize_rank(outputs, 1), targets)
        # triplet_loss = self.triplet(embeddings, targets)
        triplet_loss = self.triplet(embeddings, targets, embeddings_augment, weights)
        center_loss = self.center(embeddings, targets, embeddings_augment, weights)
        return (1. - self.circle_factor) * smooth_loss + \
               triplet_loss + \
               self.lamda * center_loss + \
               self.circle_factor * circle_loss


class RepreLoss(nn.Module):
    def __init__(self, num_classes, lamda=0.0005, margin=0.3, feat_dim=512, ckpt=None):
        super(RepreLoss, self).__init__()
        self.triplet = TripletLoss(margin, 0.0, reduction="none")
        self.center = CenterLoss(num_classes, feat_dim, ckpt=ckpt)
        self.lamda = lamda

    def forward(self, embeddings, targets, weights):
        return self.triplet(embeddings, targets, weights) + self.lamda * self.center(embeddings, targets,
                                                                                     weights=weights)
