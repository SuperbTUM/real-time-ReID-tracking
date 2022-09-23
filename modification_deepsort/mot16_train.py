import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SERes18_IBN import SEDense18_IBN

cudnn.deterministic = True
cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.1):
        """ Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        target = target.long()
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


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
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
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
            global_feat = F.normalize(global_feat, axis=-1)
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


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

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
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class HybridLoss3(nn.Module):
    def __init__(self, num_classes, feat_dim=512, margin=0.3, smoothing=0.1):
        super().__init__()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        # self.triplet = TripletLoss(margin)
        self.triplet = WeightedRegularizedTriplet()
        self.smooth = LabelSmoothing(smoothing)

    def forward(self, embeddings, outputs, targets):
        """
        features: feature vectors
        targets: ground truth labels
        """
        smooth_loss = self.smooth(outputs, targets)
        triplet_loss = self.triplet(embeddings, targets)
        center_loss = self.center(embeddings, targets)
        return smooth_loss + triplet_loss + 0.0005 * center_loss


class InVideoModel(SEDense18_IBN):
    def __init__(self, num_class, needs_norm=True, gem=True, is_reid=False):
        super(InVideoModel, self).__init__(num_class=num_class,
                                           needs_norm=needs_norm,
                                           gem=gem,
                                           is_reid=is_reid)

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)
        branch1 = x
        x = self.basicBlock11(x)
        scale1 = self.seblock1(x)
        x = scale1 * x + branch1

        branch2 = x
        x = self.basicBlock12(x)
        scale2 = self.seblock2(x)
        x = scale2 * x + branch2

        branch3 = x
        x = self.basicBlock21(x)
        scale3 = self.seblock3(x)
        if self.needs_norm:
            x = scale3 * x + self.optionalNorm2dconv3(self.ancillaryconv3(branch3))
        else:
            x = scale3 * x + self.ancillaryconv3(branch3)

        branch4 = x
        x = self.basicBlock22(x)
        scale4 = self.seblock4(x)
        x = scale4 * x + branch4

        branch5 = x
        x = self.basicBlock31(x)
        scale5 = self.seblock5(x)
        if self.needs_norm:
            x = scale5 * x + self.optionalNorm2dconv5(self.ancillaryconv5(branch5))
        else:
            x = scale5 * x + self.ancillaryconv5(branch5)

        branch6 = x
        x = self.basicBlock32(x)
        scale6 = self.seblock6(x)
        x = scale6 * x + branch6

        branch7 = x
        x = self.basicBlock41(x)
        scale7 = self.seblock7(x)
        if self.needs_norm:
            x = scale7 * x + self.optionalNorm2dconv7(self.ancillaryconv7(branch7))
        else:
            x = scale7 * x + self.ancillaryconv7(branch7)

        branch8 = x
        x = self.basicBlock42(x)
        scale8 = self.seblock8(x)
        x = scale8 * x + branch8

        bs, c, h, w = x.size()
        x = x.view(b, s, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, s * h, w)

        x = self.avgpooling(x)
        feature = x.view(x.size(0), -1)
        if self.is_reid:
            # do we need a further normalization here?
            return F.normalize(feature, p=2, dim=1)
        x = self.bnneck(feature)
        x = self.classifier(x)

        return x, feature


class VideoDataset(Dataset):
    def __init__(self, gt_paths, transforms, seq_len=10, prefix_image_path="../datasets/MOT16/train/"):
        super(VideoDataset, self).__init__()
        self.gt_paths = gt_paths
        self.seq_len = seq_len
        self.transforms = transforms
        self.gt_info, self.labels = self.read_gt()
        self.prefix_image_path = prefix_image_path

    def read_gt(self):
        gt_info = defaultdict(list)
        label = -1
        diff = 0
        labels = []
        for path in self.gt_paths:
            with open(path, "r") as f:
                lines = f.readlines()
            f.close()
            for line in lines:
                line = list(map(lambda x: float(x), line.strip().split(",")))
                if line[-2] == 1:
                    # pedestrian
                    if line[-2] == 0:
                        continue
                    if line[1] - label != diff:
                        label += 1
                        labels.append(label)
                        diff = line[1] - label
                    bbox = list(map(lambda x: float(x), line[2:6]))
                    frame = line[0]
                    file_loc = path.split("/")[-3] + "/"
                    detail = bbox + [int(frame)] + [file_loc]
                    gt_info[label].append(detail)
        return gt_info, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        cur_gt_info = self.gt_info[item]
        # take the image sequence of a person, so the item number is the person number
        length = len(cur_gt_info)
        images = list()
        if length < self.seq_len:
            # fill with last frame
            final_gt_info = cur_gt_info + [cur_gt_info[-1] for _ in range(self.seq_len - length)]
        else:
            # sample
            indices = np.random.choice(np.arange(len(cur_gt_info)), size=self.seq_len, replace=False)
            final_gt_info = []
            for indice in indices:
                final_gt_info.append(cur_gt_info[indice])
        for i, gt in enumerate(final_gt_info):
            left, top, width, height, frame, loc = gt
            absolute_loc = self.prefix_image_path + loc + "img1/" + str(frame).zfill(6) + ".jpg"
            image = Image.open(absolute_loc).convert("RGB")
            image = image.crop(
                (round(max(0, left)),
                 round(max(0, top)),
                 round(min(image.size[0], left + width)),
                 round(min(image.size[1], top + height)))
            )
            if self.transforms:
                image = self.transforms(image)
            images.append(image)
        images = torch.stack(images)
        label = torch.tensor(self.labels[item]).int()
        return images, label


def train(dataset, batch_size=8, epochs=25, num_classes=517):
    model = InVideoModel(num_class=num_classes, gem=False).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func = HybridLoss3(num_classes=num_classes)
    loss_stats = []
    for epoch in range(epochs):
        dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample
            optimizer.zero_grad()
            images = images.cuda()
            label = label.cuda()
            prediction, feature = model(images)
            loss = loss_func(feature, prediction, label)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "in_video_checkpoint.pt")
    return model, loss_stats

# What if the dataset is too large or computing device is not strong enough?
# Option 1: distributed training
def distributed_train(dataset, batch_size=8, epochs=25, num_classes=517,
                      rank=-1, world_size=-1):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    import torch.distributed as dist
    assert dist.is_available()
    from torch.nn.parallel import DistributedDataParallel as DDP

    def setup(rank=rank, world_size=world_size):
        if dist.is_nccl_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    setup(rank, world_size)

    """Original training
    """
    model = InVideoModel(num_class=num_classes, gem=False).cuda()
    # Distributed training
    # ----------
    from torch.utils.data.distributed import DistributedSampler
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.train()
    # ----------
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func = HybridLoss3(num_classes=num_classes)
    loss_stats = []
    for epoch in range(epochs):
        dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True, sampler=DistributedSampler(dataset))
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample
            optimizer.zero_grad()
            images = images.cuda()
            label = label.cuda()
            prediction, feature = ddp_model(images)
            loss = loss_func(feature, prediction, label)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(ddp_model.parameters(), 10)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    ddp_model.eval()
    if rank == 0:
        torch.save(ddp_model.state_dict(), "in_video_checkpoint.pt")
    dist.barrier()
    cleanup()
    # Could initialize the job with torchrun
    return ddp_model, loss_stats


def activate_distributed_train(train_fn, world_size):
    import torch.multiprocessing as mp
    mp.spawn(train_fn,
             args=(world_size, ),
             nprocs=world_size,
             join=True)


def distributed_demo(dataset, batch_size=8, epochs=25, num_classes=517,
                     rank=0, world_size=1):
    activate_distributed_train(distributed_train(dataset, batch_size, epochs, num_classes,
                                                 rank, world_size), world_size)


def plot_loss(loss_stats):
    plt.figure()
    plt.plot(loss_stats, linewidth=2, label="train loss")
    plt.xlabel("iterations")
    plt.ylabel('loss')
    plt.title('training loss')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train_index = [2, 4, 5, 9, 10, 11, 13]
    gt_paths = ["../datasets/MOT16/train/MOT16-" + str(i).zfill(2) + "/gt/gt.txt" for i in train_index]
    image_paths = ["../datasets/MOT16/train/MOT16-" + str(i).zfill(2) + "/*.jpg" for i in train_index]
    gt_images = []
    for image_path in image_paths:
        gt_images.extend(sorted(glob.glob(image_path)))
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])
    dataset = VideoDataset(gt_paths, transform)
    model, loss_stats = train(dataset, num_classes=len(dataset))
    plot_loss(loss_stats)
