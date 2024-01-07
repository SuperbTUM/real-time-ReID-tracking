import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn, autograd


def get_feats(model, dataloader):
    model.eval()
    labels = []
    features = []
    cams = []
    for sample in dataloader:
        image, label, cam = sample[:3]
        feat = model(image.cuda(), cam)
        features.append(feat[1])
        labels.append(label)
        cams.append(cam)
    labels = torch.cat(labels, dim=0)
    features = torch.cat(features, dim=0)
    cams = torch.cat(cams, dim=0)
    model.train()
    return labels, features, cams


# from https://github.com/htyao89/Cross-View-Asymmetric-Cluster-Contrastive/blob/main/examples/main.py
class DCC(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut_ccc, lut_icc,  momentum):
        ctx.lut_ccc = lut_ccc
        ctx.lut_icc = lut_icc
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_ccc = inputs.mm(ctx.lut_ccc.t())
        outputs_icc = inputs.mm(ctx.lut_icc.t())

        return outputs_ccc,outputs_icc

    @staticmethod
    def backward(ctx, grad_outputs_ccc, grad_outputs_icc):
        inputs,targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs_ccc.mm(ctx.lut_ccc)+grad_outputs_icc.mm(ctx.lut_icc)

        batch_centers = defaultdict(list)
        for instance_feature, index in zip(inputs, targets.data.cpu().numpy()):
            batch_centers[index].append(instance_feature)

        for y, features in batch_centers.items():
            mean_feature = torch.stack(batch_centers[y],dim=0)
            non_mean_feature = mean_feature.mean(0)
            x = F.normalize(non_mean_feature,dim=0)
            ctx.lut_ccc[y] = ctx.momentum * ctx.lut_ccc[y] + (1.-ctx.momentum) * x
            ctx.lut_ccc[y] /= ctx.lut_ccc[y].norm()

        del batch_centers

        for x, y in zip(inputs,targets.data.cpu().numpy()):
            ctx.lut_icc[y] = ctx.lut_icc[y] * ctx.momentum + (1 - ctx.momentum) * x
            ctx.lut_icc[y] /= ctx.lut_icc[y].norm()

        return grad_inputs, None, None, None, None


def oim(inputs, targets, lut_ccc, lut_icc, momentum=0.1):
    return DCC.apply(inputs, targets, lut_ccc, lut_icc, torch.Tensor([momentum]).to(inputs.device))


class DCCLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.1,
                 weight=0.25, size_average=True):
        super(DCCLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut_ccc', torch.zeros(num_classes, num_features).cuda())
        self.register_buffer('lut_icc', torch.zeros(num_classes, num_features).cuda())

        print('Weight:{}, Momentum:{}'.format(self.weight,self.momentum))

    def forward(self, inputs, targets):
        inputs_ccc, inputs_icc = oim(inputs, targets, self.lut_ccc, self.lut_icc, momentum=self.momentum)

        inputs_ccc *= self.scalar
        inputs_icc *= self.scalar

        loss_ccc = F.cross_entropy(inputs_ccc, targets, size_average=self.size_average, label_smoothing=0.1)
        loss_icc = F.cross_entropy(inputs_icc, targets, size_average=self.size_average, label_smoothing=0.1)

        '''
        targets = F.one_hot(targets, inputs.size(1))
        probs_ccc = F.softmax(inputs_ccc, dim=1)
        one_minus_pt_ccc = torch.sum(targets * (1 - probs_ccc), dim=1)
        loss_ccc += one_minus_pt_ccc.mean()

        probs_icc = F.softmax(inputs_icc, dim=1)
        one_minus_pt_icc = torch.sum(targets * (1 - probs_icc), dim=1)
        loss_icc += one_minus_pt_icc.mean()
        '''

        loss_con = F.smooth_l1_loss(inputs_ccc, inputs_icc.detach(), reduction='elementwise_mean')
        loss = loss_ccc+loss_icc+self.weight*loss_con

        return loss


def generate_centers(model, dataloader):
    labels, features = get_feats(model, dataloader)[:2]
    centers = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i].item()].append(features[i])

    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]

    centers = torch.stack(centers, dim=0)
    return centers
