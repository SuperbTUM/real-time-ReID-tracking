from train_prepare import *
import torch.onnx
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable

from plr_osnet import plr_osnet
from vision_transformer import vit_t, swin_t
from train_utils import *
from dataset_market import Market1501

import argparse


class MarketDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        return detailed_info


def train_plr_osnet(dataset, batch_size=8, epochs=25, num_classes=517):
    model = plr_osnet(num_classes=num_classes, loss='triplet').cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func1 = HybridLoss3(num_classes=num_classes, feat_dim=2048)
    loss_func2 = HybridLoss3(num_classes=num_classes, feat_dim=512)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    loss_stats = []
    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample[:2]
            optimizer.zero_grad()
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            global_branch, local_branch, feat = model(images)
            loss1 = loss_func1(feat[0], global_branch, label)
            loss2 = loss_func2(feat[1], local_branch, label)
            loss = loss1 + loss2
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "plr_osnet_checkpoint.pt")
    to_onnx(model, torch.randn(batch_size, 3, 256, 128, requires_grad=True))
    return model, loss_stats


def train_vision_transformer(dataset, backbone, batch_size=8, epochs=25, num_classes=517, all_cams=6, all_seq=6):
    if backbone == "vit":
        model = vit_t(img_size=(448, 224), num_classes=num_classes, loss="triplet", camera=all_cams, sequence=all_seq, side_info=True).cuda()
    elif backbone == "swin":
        model = swin_t(num_classes=num_classes, loss="triplet", camera=all_cams, sequence=all_seq, side_info=True).cuda()
    else:
        raise NotImplementedError
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func = HybridLoss3(num_classes=num_classes, feat_dim=384)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    loss_stats = []
    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            if len(sample) == 2:
                images, label = sample
                view_index = None
            elif len(sample) == 3:
                images, label, cam = sample
                view_index = cam
            else:
                images, label, cam, seqid = sample
                view_index = cam * all_cams + seqid
            images = images.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            logits, embeddings = model(images, view_index)
            loss = loss_func(embeddings, logits, label)
            loss_stats.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "vision_transformer_checkpoint.pt")
    to_onnx(model, torch.randn(batch_size, 3, 448, 224, requires_grad=True))
    return model, loss_stats


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="plr_osnet", choices=["plr_osnet", "vit", "swin"])
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--accelerate", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    dataset = Market1501(root="Market1501")

    if params.backbone == "plr_osnet":
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            LGT(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        market_dataset = MarketDataset(dataset.train, transform)
        model, loss_stats = train_plr_osnet(market_dataset, params.bs, params.epochs, dataset.num_train_pids)
    else:
        transform = transforms.Compose([
            transforms.Resize((448, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((448, 224)),
            LGT(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        market_dataset = MarketDataset(dataset.train, transform)
        model, loss_stats = train_vision_transformer(market_dataset, params.backbone, params.bs, params.epochs,
                                                     dataset.num_train_pids,
                                                     dataset.num_train_cams,
                                                     dataset.num_train_seqs)

    print("loss curve", loss_stats)
