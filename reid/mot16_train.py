from train_prepare import *
import torch.onnx
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision import transforms
import math
import glob

from SERes18_IBN import SEDense18_IBN
from plr_osnet import plr_osnet

cudnn.deterministic = True
cudnn.benchmark = True


def train_plr_osnet(dataset, batch_size=8, epochs=25, num_classes=517):
    model = plr_osnet(num_classes=num_classes, loss='triplet').cuda()
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
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            global_branch, local_branch, feat = model(images)
            loss1 = loss_func(feat[:2048], global_branch, label)
            loss2 = loss_func(feat[2048:], local_branch, label)
            loss = loss1 + loss2
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "in_video_checkpoint.pt")
    to_onnx(model, torch.randn(batch_size, 3, 256, 128, requires_grad=True))
    return model, loss_stats


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

import h5py
import cv2
def transform_dataset_hdf5(gt_paths, img_width, img_height):
    """With OpenCV"""
    h5file = "import_images.h5"
    with h5py.File(h5file, "w") as h5f:
        image_ds = h5f.create_dataset("images", shape=(len(gt_paths), img_width, img_height, 3), dtype=int)
        for cnt, path in enumerate(gt_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img, (img_width, img_height))
            image_ds[cnt:cnt+1, :, :] = img_resize
    return image_ds

def recover_from_hdf5(image_ds, index):
    with h5py.File(image_ds, "r") as h5f:
        # Random access
        image = h5f["images"][index, ...]
    return image


class VideoDataset(Dataset):
    def __init__(self, gt_paths, transforms, seq_len=10, prefix_image_path="../datasets/MOT16/train/"):
        super(VideoDataset, self).__init__()
        """Make the data loading lighter with h5py?"""
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
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
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
        dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset))
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample
            optimizer.zero_grad()
            images = images.cuda()
            label = label.cuda()
            prediction, feature = ddp_model(images)
            loss = loss_func(feature, prediction, label)
            # ------
            dist.all_reduce(loss.clone(), op=dist.ReduceOp.SUM)
            loss_record = loss / world_size
            # ------
            loss_stats.append(loss_record.cpu().item())
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

# This is the code of Local Grayscale Transfomation

class LGT(object):

    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        new = img.convert("L")   # Convert from here to the corresponding grayscale image
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return img

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

                return img

        return img


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
        LGT(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])
    dataset = VideoDataset(gt_paths, transform)
    model, loss_stats = train(dataset, num_classes=len(dataset))
    plot_loss(loss_stats)
