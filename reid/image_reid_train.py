from train_prepare import *
import os
import torch.onnx
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable

from plr_osnet import plr_osnet
from vision_transformer import vit_t
from swin_transformer import swin_t
from train_utils import *
from dataset_market import Market1501

import argparse


cudnn.deterministic = True
cudnn.benchmark = True


class MarketDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        self.images_pseudo = []
        self._continual = False

    def set_cross_domain(self):
        self._continual = True

    def reset_cross_domain(self):
        self._continual = False

    def __len__(self):
        if self._continual:
            return len(self.images_pseudo) + len(self.images)
        return len(self.images)

    def add_pseudo(self, pseudo_labeled_data):
        self.images_pseudo.extend(pseudo_labeled_data)

    def __getitem__(self, item):
        if self._continual:
            if item < len(self.images):
                detailed_info = list(self.images[item])
            else:
                detailed_info = list(self.images_pseudo[item - len(self.images)])
        else:
            detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        for i in range(2, len(detailed_info)):
            detailed_info[i] = torch.tensor(detailed_info[i], dtype=torch.long)
        # if self._continual:
        #     return detailed_info + [0 if item < len(self.images) else 1]
        return detailed_info


def train_plr_osnet(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func1 = HybridLoss3(num_classes=num_classes, feat_dim=2048)
    loss_func2 = HybridLoss3(num_classes=num_classes, feat_dim=512)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    if accelerate:
        res_dict = accelerate_train(model, dataloader, optimizer, lr_scheduler)
        model, dataloader, optimizer, lr_scheduler = res_dict["accelerated"]
        accelerator = res_dict["accelerator"]
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
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "plr_osnet_checkpoint.pt")
    to_onnx(model,
            torch.randn(batch_size, 3, 256, 128, requires_grad=True, device="cuda"),
            output_names=["y1", "y2", "fea"])
    return model, loss_stats


def train_vision_transformer(model, dataset, feat_dim=384, batch_size=8, epochs=25, num_classes=517,
                             all_cams=6, all_seq=6, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func = HybridLoss3(num_classes=num_classes, feat_dim=feat_dim)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    if accelerate:
        res_dict = accelerate_train(model, dataloader, optimizer, lr_scheduler)
        model, dataloader, optimizer, lr_scheduler = res_dict["accelerated"]
        accelerator = res_dict["accelerator"]
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
                assert all(cam) < all_cams
                view_index = cam
            else:
                images, label, cam, seqid = sample
                assert all(seqid) < all_seq and all(cam) < all_cams
                if cam is not None and any(cam) >= 0 and seqid is not None and any(seqid) >= 0:
                    view_index = cam * all_cams + seqid
                else:
                    view_index = None
            images = images.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            logits, embeddings = model(images, view_index)
            loss = loss_func(embeddings, logits, label)
            loss_stats.append(loss.cpu().item())
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "vision_transformer_checkpoint.pt")
    to_onnx(model,
            (torch.randn(batch_size, 3, 448, 224, requires_grad=True, device="cuda"),
             torch.ones(batch_size, dtype=torch.long)),
            input_names=["input", "index"],
            output_names=["embedding", "output"])
    return model, loss_stats


def side_info_only(model):
    model.train()
    for name, params in model.named_parameters():
        if name.endswith("side_info_embedding"):
            params.requires_grad = True
        else:
            params.requires_grad = False
    return model


def inference(model, dataloader, all_cam=6, conf_thres=0.7, use_onnx=False) -> list:
    model.eval()
    labels = []
    with torch.no_grad():
        for sample in dataloader:
            if len(sample) == 2:
                img, _ = sample
                cam = seq = None
            elif len(sample) == 3:
                img, _, cam = sample
                seq = None
            else:
                img, _, cam, seq = sample
            img = img.cuda(non_blocking=True)
            if cam is not None and any(cam) >= 0 and seq is not None and any(seq) >= 0:
                _, preds = model(img, cam * all_cam + seq)
            else:
                _, preds = model(img)
            preds = preds.squeeze()
            preds = preds.softmax(dim=-1)
            conf = preds.max(dim=-1)
            candidates = preds.argmax(dim=-1)
            for i, c in enumerate(conf):
                if c > conf_thres:
                    labels.append((img[i], candidates[i], cam[i] if cam else None, seq[i] if seq else None))
    return labels


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt", help="where the checkpoint of vit is, can either be a onnx or pt", type=str,
                      default="vision_transformer_checkpoint.pt")
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="plr_osnet", choices=["plr_osnet", "vit", "swin_v1", "swin_v2"])
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--continual", action="store_true")
    args.add_argument("--accelerate", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    dataset = Market1501(root="Market1501")

    if params.backbone == "plr_osnet":
        # No need for cross-domain retrain
        transform_train = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            LGT(),
            # transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        market_dataset = MarketDataset(dataset.train, transform_train)
        model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
        model, loss_stats = train_plr_osnet(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                            params.accelerate)

        if params.continual:
            transform_test = transforms.Compose([transforms.Resize((256, 128)),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                                 transforms.ToTensor(),
                                                 ]
                                                )
            # This is fake
            dataset_test = MarketDataset(dataset.gallery, transform_test)
            dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4, pin_memory=True)

            pseudo_labeled_data = inference(model, dataloader_test)
            del dataset_test
            market_dataset.add_pseudo(pseudo_labeled_data)
            market_dataset.set_cross_domain()
            model = side_info_only(model)
            model, loss_stats = train_plr_osnet(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                                params.accelerate)
            market_dataset.reset_cross_domain()

    else:
        transform_train = transforms.Compose([
            transforms.Resize((448, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((448, 224)),
            LGT(),
            # transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             transforms.ToTensor(),
                                             ]
                                            )
        market_dataset = MarketDataset(dataset.train, transform_train)

        if params.backbone.startswith("vit"):
            model = vit_t(img_size=(448, 224), num_classes=dataset.num_train_pids, loss="triplet",
                          camera=dataset.num_train_cams,
                          sequence=dataset.num_train_seqs, side_info=True).cuda()
            model, loss_stats = train_vision_transformer(model, market_dataset, 384,
                                                         params.bs, params.epochs,
                                                         dataset.num_train_pids,
                                                         dataset.num_train_cams,
                                                         dataset.num_train_seqs,
                                                         params.accelerate)
            if params.continual:
                dataset_test = MarketDataset(dataset.gallery, transform_test)
                dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4,
                                              pin_memory=True)

                pseudo_labeled_data = inference(model, dataloader_test, dataset.num_train_cams)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data)
                market_dataset.set_cross_domain()
                model, loss_stats = train_vision_transformer(model, market_dataset, 384,
                                                             params.bs, params.epochs,
                                                             dataset.num_train_pids,
                                                             dataset.num_train_cams,
                                                             dataset.num_train_seqs)
                market_dataset.reset_cross_domain()
        elif params.backbone.startswith("swin"):
            model = swin_t(num_classes=dataset.num_train_pids, loss="triplet",
                           camera=dataset.num_train_cams, sequence=dataset.num_train_seqs,
                           side_info=True,
                           version=params.backbone[-2:]).cuda()
            model, loss_stats = train_vision_transformer(model, market_dataset, 96,
                                                         params.bs, params.epochs,
                                                         dataset.num_train_pids,
                                                         dataset.num_train_cams,
                                                         dataset.num_train_seqs,
                                                         params.accelerate)

            if params.continual:
                dataset_test = MarketDataset(dataset.gallery, transform_test)
                dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4,
                                              pin_memory=True)

                pseudo_labeled_data = inference(model, dataloader_test, dataset.num_train_cams)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data)
                market_dataset.set_cross_domain()
                model, loss_stats = train_vision_transformer(model, market_dataset, 96,
                                                             params.bs, params.epochs,
                                                             dataset.num_train_pids,
                                                             dataset.num_train_cams,
                                                             dataset.num_train_seqs)
                market_dataset.reset_cross_domain()
        else:
            raise NotImplementedError

    print("loss curve", loss_stats)
