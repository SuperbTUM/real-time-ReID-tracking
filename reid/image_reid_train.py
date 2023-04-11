import torch.onnx
from torch.utils.data import Dataset
from tqdm import tqdm

from backbones.baseline_lite import ft_baseline
from backbones.plr_osnet import plr_osnet
from backbones.SERes18_IBN import seres18_ibn
from backbones.vision_transformer import vit_t
from backbones.swin_transformer import swin_t
from train_utils import *
from dataset_market import Market1501
from train_prepare import WarmupMultiStepLR

import argparse
import onnxruntime
from scipy.special import softmax
import madgrad


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


def train_cnn(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)# WarmupMultiStepLR(optimizer, [10, 30])
    loss_func = HybridLoss(num_classes, 512, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
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
            embeddings, outputs = model(images)
            loss = loss_func(embeddings, outputs, label)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.eval()
    try:
        to_onnx(model.module,
                torch.randn(1, 3, 256, 128, requires_grad=True, device="cuda"),
                output_names=["embeddings", "outputs"])
    except RuntimeError:
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint.pt")
    return model, loss_stats


def train_plr_osnet(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    optimizer = madgrad.MADGRAD(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = WarmupMultiStepLR(optimizer, [10, 30])
    loss_func1 = HybridLoss(num_classes, 2048, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    loss_func2 = HybridLoss(num_classes, 512, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
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
            loss = loss1 + loss2  # / 2.0
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.eval()
    to_onnx(model.module,
            torch.randn(1, 3, 256, 128, requires_grad=True, device="cuda"),
            output_names=["y1", "y2", "fea"])
    torch.save(model.state_dict(), "checkpoint/plr_osnet_checkpoint.pt")
    return model, loss_stats


def train_vision_transformer(model, dataset, feat_dim=384, batch_size=8, epochs=25, num_classes=517,
                             all_cams=6, all_seq=6, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = WarmupMultiStepLR(optimizer, [10, 30])
    loss_func = HybridLoss(num_classes, feat_dim, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
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
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.eval()
    to_onnx(model.module,
            (torch.randn(1, 3, 448, 224, requires_grad=True, device="cuda"),
             torch.ones(1, dtype=torch.long)),
            input_names=["input", "index"],
            output_names=["embeddings", "outputs"])
    torch.save(model.state_dict(), "checkpoint/vision_transformer_checkpoint.pt")
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
    if not use_onnx:
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
                preds = preds.softmax(dim=-1)
                conf = preds.max(dim=-1)
                candidates = preds.argmax(dim=-1).cpu().numpy()
                cam = cam.cpu().numpy()
                seq = seq.cpu().numpy()
                for i, c in enumerate(conf):
                    if c > conf_thres:
                        labels.append((img[i], candidates[i], cam[i] if cam else None, seq[i] if seq else None))
    else:
        for sample in dataloader:
            if len(sample) == 2:
                img, _ = sample
                cam = seq = None
            elif len(sample) == 3:
                img, _, cam = sample
                seq = None
            else:
                img, _, cam, seq = sample
            ort_inputs = {'input': to_numpy(img),
                          "index": to_numpy(cam * all_cam + seq)}
            preds = ort_session.run(["embeddings", "outputs"], ort_inputs)[1]
            preds = softmax(preds, axis=-1)
            conf = preds.max(axis=-1)
            candidates = preds.argmax(axis=-1)
            for i, c in enumerate(conf):
                if c > conf_thres:
                    labels.append((img[i], candidates[i], cam[i] if cam else None, seq[i] if seq else None))
    return labels


def parser():
    def range_type(astr, min=-1., max=1.):
        value = float(astr)
        if min <= value <= max:
            return value
        else:
            raise argparse.ArgumentTypeError('value not in range %s-%s' % (min, max))
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default="~/real-time-ReID-tracking")
    args.add_argument("--ckpt", help="where the checkpoint of vit is, can either be a onnx or pt", type=str,
                      default="vision_transformer_checkpoint.pt")
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="plr_osnet", choices=["seres18",
                                                                            "plr_osnet",
                                                                            "vit",
                                                                            "swin_v1",
                                                                            "swin_v2",
                                                                            "baseline"])
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--epsilon", help="for polyloss, 0 by default", type=range_type, default=0.0, metavar="[-1, 1]")
    args.add_argument("--margin", help="for triplet loss", default=0.0, type=float)
    args.add_argument("--center_lamda", help="for center loss", default=0.0, type=float)
    args.add_argument("--continual", action="store_true")
    args.add_argument("--accelerate", action="store_true")
    args.add_argument("--renorm", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    dataset = Market1501(root="/".join((params.root, "Market1501")))

    if params.backbone in ("plr_osnet", "seres18", "baseline"):
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
        if params.backbone == "plr_osnet":
            model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_plr_osnet(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                                params.accelerate)
        else:
            if params.backbone == "seres18":
                model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=params.renorm).cuda()
            else:
                model = ft_baseline(dataset.num_train_pids).cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_cnn(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                          params.accelerate)

        # if params.continual:
        #     transform_test = transforms.Compose([transforms.Resize((256, 128)),
        #                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #                                          transforms.ToTensor(),
        #                                          ]
        #                                         )
        #     dataset_test = MarketDataset(dataset.gallery, transform_test)
        #     dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4, pin_memory=True)
        #
        #     pseudo_labeled_data = inference(model, dataloader_test)
        #     del dataset_test
        #     market_dataset.add_pseudo(pseudo_labeled_data)
        #     market_dataset.set_cross_domain()
        #     model = side_info_only(model)
        #     model, loss_stats = train_plr_osnet(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
        #                                         params.accelerate)
        #     market_dataset.reset_cross_domain()

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
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        market_dataset = MarketDataset(dataset.train, transform_train)
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

        if params.backbone.startswith("vit"):
            model = vit_t(img_size=(448, 224), num_classes=dataset.num_train_pids, loss="triplet",
                          camera=dataset.num_train_cams,
                          sequence=dataset.num_train_seqs, side_info=True).cuda()
            model = nn.DataParallel(model)
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

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model.onnx", providers=providers)

                pseudo_labeled_data = inference(model, dataloader_test, dataset.num_train_cams, use_onnx=True)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data)
                market_dataset.set_cross_domain()
                model = side_info_only(model)
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
            model = nn.DataParallel(model)
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

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model.onnx", providers=providers)
                pseudo_labeled_data = inference(model, dataloader_test, dataset.num_train_cams, use_onnx=True)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data)
                market_dataset.set_cross_domain()
                model = side_info_only(model)
                model, loss_stats = train_vision_transformer(model, market_dataset, 96,
                                                             params.bs, params.epochs,
                                                             dataset.num_train_pids,
                                                             dataset.num_train_cams,
                                                             dataset.num_train_seqs)
                market_dataset.reset_cross_domain()
        else:
            raise NotImplementedError

    plot_loss(loss_stats)
