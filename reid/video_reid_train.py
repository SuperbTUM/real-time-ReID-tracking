from train_prepare import *
import torch.onnx
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision import transforms
import glob

from video_model import InVideoModel
from train_utils import *

cudnn.deterministic = True
cudnn.benchmark = True


class VideoDataset(Dataset):
    def __init__(self, gt_paths, transforms, lamda=1.0, seq_len=10, prefix_image_path="../datasets/MOT16/train/"):
        super(VideoDataset, self).__init__()
        """Make the data loading lighter with h5py?"""
        assert lamda >= 1.0
        self.gt_paths = gt_paths
        self.seq_len = seq_len
        self.transforms = transforms
        self.gt_info, self.labels = self.read_gt()
        self.prefix_image_path = prefix_image_path
        self.lamda = lamda

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
                    if self.lamda > 1.0:
                        bbox = [max(0., bbox[0]-bbox[0]*(self.lamda - 1)/2), max(0., bbox[1]-bbox[1]*(self.lamda - 1)/2),
                                bbox[2]*self.lamda, bbox[3]*self.lamda]
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
                # Random Augment
                random.shuffle(self.transforms)
                transform_final = transforms.Compose(self.transforms)
                image = transform_final(image)
            image = transforms.ToTensor()(image)
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
    model = InVideoModel(num_class=num_classes, gem=False).cuda()
    ddp_model = ddp_trigger(model, rank, world_size)
    """Original training
    """
    # Distributed training
    # ----------
    from torch.utils.data.distributed import DistributedSampler
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
    postprocess_ddp(ddp_model, rank)
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
    transform_candidates = [
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        LGT(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # transforms.ToTensor(),
    ]
    dataset = VideoDataset(gt_paths, transform_candidates, 1.0)
    model, loss_stats = train(dataset, num_classes=len(dataset))
    plot_loss(loss_stats)