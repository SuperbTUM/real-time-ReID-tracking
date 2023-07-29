import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from backbones.baseline_lite import ft_baseline
from backbones.plr_osnet import plr_osnet
from backbones.SERes18_IBN import seres18_ibn
from backbones.CARes18 import cares18_ibn
from backbones.vision_transformer import vit_t
from backbones.swin_transformer import swin_t
from train_utils import *
from dataset_market import Market1501
from dataset_dukemtmc import DukeMTMCreID
from train_prepare import WarmupMultiStepLR, RandomIdentitySampler, RandomErasing

import argparse
import onnxruntime
import madgrad
from sklearn.cluster import DBSCAN
from accelerate import Accelerator

cudnn.deterministic = True
cudnn.benchmark = True

# assert export_yolo()

class reidDataset(Dataset):
    def __init__(self, images, train_classes, transform=None, get_crop=False):
        self.images = images
        self.train_classes = train_classes
        self.transform = transform
        self.images_pseudo = []
        self._continual = False
        self.cropped = []
        self.cropped_pseudo = []
        self.class_stats = [0 for _ in range(train_classes)]
        for image in images:
            if image[1] < train_classes:
                self.class_stats[image[1]] += 1
        self.get_crop = get_crop
        if get_crop:
            pure_images = list(map(lambda x: x[0], images))
            i = 0
            while i < len(pure_images):
                local_batch = []
                end = min(i+64, len(pure_images))
                for j in range(i, end):
                    local_batch.append(Image.open(pure_images[j]).convert("RGB"))
                cropped_imgs = recrop(local_batch, "pil")
                self.cropped.extend(cropped_imgs)
                i = end

    def get_class_stats(self):
        return self.class_stats

    def set_cross_domain(self):
        self._continual = True

    def reset_cross_domain(self):
        self._continual = False

    def __len__(self):
        if self._continual:
            return len(self.images_pseudo) + len(self.images)
        return len(self.images)

    def add_pseudo(self, pseudo_labeled_data, num_class_new):
        self.images_pseudo.extend(pseudo_labeled_data)
        self.class_stats = self.class_stats + [0 for _ in range(num_class_new - self.train_classes)]
        for image in self.images_pseudo:
            if image[1] >= self.train_classes:
                self.class_stats[image[1]] += 1
        if self.get_crop:
            pure_images = list(map(lambda x: x[0], self.images_pseudo))
            i = 0
            while i < len(pure_images):
                local_batch = []
                end = min(i + 64, len(pure_images))
                for j in range(i, end):
                    local_batch.append(Image.open(pure_images[j]).convert("RGB"))
                cropped_imgs = recrop(local_batch, "pil")
                self.cropped_pseudo.extend(cropped_imgs)
                i = end

    def __getitem__(self, item):
        if self._continual:
            if item < len(self.images):
                detailed_info = list(self.images[item])
            else:
                detailed_info = list(self.images_pseudo[item - len(self.images)])
        else:
            detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        # if self.get_crop and np.random.random() > 0.5:
        #     if item < len(self.images):
        #         detailed_info[0] = self.cropped[item]
        #     else:
        #         detailed_info[0] = self.cropped_pseudo[item - len(self.images)]
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        for i in range(2, len(detailed_info)):
            detailed_info[i] = torch.tensor(detailed_info[i], dtype=torch.long)
        if self._continual:
            return detailed_info + [1. if item < len(self.images) else 2.] # tricky
        return detailed_info


def train_cnn(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    class_stats = dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1./stat) for stat in class_stats])).cuda() * num_classes
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    loss_func = HybridLoss(num_classes, 512, params.margin, epsilon=params.epsilon, lamda=params.center_lamda, class_stats=class_stats)
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)

    if params.instance > 0:
        custom_sampler = RandomIdentitySampler(dataset, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70],
                                     gamma=0.1)  # WarmupMultiStepLR(optimizer, [10, 30])
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=not params.instance,
                             pin_memory=True, sampler=custom_sampler)
    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, lr_scheduler, optimizer_center = accelerator.prepare(model, dataloader, optimizer, lr_scheduler, optimizer_center)
    loss_stats = []
    # transforms_augment = nn.Sequential(transforms.RandomHorizontalFlip(p=1))
    # scripted_transforms_augment = torch.jit.script(transforms_augment).cuda()
    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label, cams = sample[:3]
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            # images_flip = scripted_transforms_augment(images)
            label = Variable(label).cuda(non_blocking=True)
            # cams = cams.cuda(non_blocking=True)
            embeddings, outputs = model(images)#, cams)
            # embeddings_augment, _ = model(images_flip)
            loss = loss_func(embeddings, outputs, label)
            # loss = loss_func(embeddings, outputs, label, embeddings_augment)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            for param in loss_func.center.parameters():
                param.grad.data *= (1. / params.center_lamda)
            optimizer_center.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.needs_norm = True
    model.eval()
    loss_func.center.save()
    try:
        to_onnx(model.module,
                torch.randn(2, 3, 256, 128, requires_grad=True, device="cuda"), # bs=2, experimental
                # torch.ones(1, dtype=torch.long)),
                # input_names=["input", "index"],
                output_names=["embeddings", "outputs"])
    except: # There may be op issue
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
    lr_scheduler = WarmupMultiStepLR(optimizer, [30, 55])
    loss_func1 = HybridLoss(num_classes, 2048, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    loss_func2 = HybridLoss(num_classes, 512, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    optimizer_center1 = torch.optim.SGD(loss_func1.center.parameters(), lr=0.5)
    optimizer_center2 = torch.optim.SGD(loss_func2.center.parameters(), lr=0.5)

    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, lr_scheduler, optimizer_center1, optimizer_center2 = accelerator.prepare(
            model, dataloader, optimizer,
            lr_scheduler,
            optimizer_center1,
            optimizer_center2)
    loss_stats = []
    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample[:2]
            optimizer.zero_grad()
            optimizer_center1.zero_grad()
            optimizer_center2.zero_grad()
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
            for param in loss_func1.center.parameters():
                param.grad.data *= (1. / params.center_lamda)
            optimizer_center1.step()
            for param in loss_func2.center.parameters():
                param.grad.data *= (1. / params.center_lamda)
            optimizer_center2.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.eval()
    loss_func1.center.save()
    loss_func2.center.save()
    to_onnx(model.module,
            torch.randn(2, 3, 256, 128, requires_grad=True, device="cuda"),
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
    lr_scheduler = WarmupMultiStepLR(optimizer, [30, 55])
    loss_func = HybridLoss(num_classes, feat_dim, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)
    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, lr_scheduler, optimizer_center = accelerator.prepare(model, dataloader, optimizer,
                                                                                           lr_scheduler,
                                                                                           optimizer_center)
    loss_stats = []
    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            optimizer_center.zero_grad()
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
            for param in loss_func.center.parameters():
                param.grad.data *= (1. / params.center_lamda)
            optimizer_center.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
        lr_scheduler.step()
    model.eval()
    loss_func.center.save()
    to_onnx(model.module,
            (torch.randn(2, 3, 448, 224, requires_grad=True, device="cuda"),
             torch.ones(2, dtype=torch.long)),
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


def representation_only(model):
    # for seres18/cares18
    model.train()
    model.module.conv0.requires_grad_ = False
    model.module.bn0.requires_grad_ = False
    model.module.relu0.requires_grad_ = False
    model.module.pooling0.requires_grad_ = False
    model.module.basicBlock11.requires_grad_ = False
    model.module.basicBlock12.requires_grad_ = False
    model.module.basicBlock21.requires_grad_ = False
    model.module.basicBlock22.requires_grad_ = False
    # model.module.basicBlock31.requires_grad_ = False # tricky
    # model.module.basicBlock32.requires_grad_ = False # tricky

    # model.module.classifier.requires_grad_ = False
    # model.module.bnneck.requires_grad_ = False
    return model


def produce_pseudo_data(model, dataset_test, all_cam=6, use_onnx=False, use_side=False) -> tuple:
    model.eval()
    pseudo_data = []
    embeddings = []
    cams = []
    seqs = []
    if not use_onnx:
        dataloader = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        with torch.no_grad():
            for iteration, sample in enumerate(dataloader, 0):
                if len(sample) == 2:
                    img, _ = sample
                    cam = seq = None
                elif len(sample) == 3:
                    img, _, cam = sample
                    seq = 0
                else:
                    img, _, cam, seq = sample
                img = img.cuda(non_blocking=True)
                if use_side:
                    embedding, _ = model(img, cam + all_cam * seq)
                else:
                    embedding, _ = model(img)
                embeddings.append(embedding)
                cams.append(cam)
                seqs.append(seq)
    else:
        # experiment
        dataloader = DataLoaderX(dataset_test, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        for iteration, sample in enumerate(dataloader, 0):
            if len(sample) == 2:
                img, _ = sample
                cam = seq = None
            elif len(sample) == 3:
                img, _, cam = sample
                seq = 0
            else:
                img, _, cam, seq = sample
            if use_side:
                ort_inputs = {'input': to_numpy(img),
                              "index": to_numpy(cam + all_cam * seq)}
            else:
                ort_inputs = {'input': to_numpy(img)}
            embedding = ort_session.run(["embeddings", "outputs"], ort_inputs)[0]
            embeddings.append(torch.from_numpy(embedding))
            cams.append(cam)
            seqs.append(seq)
    embeddings = F.normalize(torch.cat(embeddings, dim=0), dim=1, p=2)
    dists = euclidean_dist(embeddings, embeddings)
    cluster_method = DBSCAN(eps=0.25, min_samples=6, metric="precomputed", n_jobs=-1)
    labels = cluster_method.fit_predict(dists)
    cams = torch.cat(cams, dim=0)
    seqs = torch.cat(seqs, dim=0)
    for i, label in enumerate(labels):
        if label != -1:
            pseudo_data.append((
                dataset_test[i][0],
                label + dataset.num_train_pids,
                cams[i].item(),
                seqs[i].item()
            ))
    print("Inference completed! {} more pseudo-labels obtained!".format(len(pseudo_data)))
    return pseudo_data, max(labels) + 1 + dataset.num_train_pids


def train_cnn_continual(model, dataset, num_class_new, batch_size=8, accelerate=False, tmp_feat_dim=256):
    model.train()
    model.needs_norm = False
    model.module.classifier[-1] = nn.Linear(tmp_feat_dim, num_class_new, bias=False)
    nn.init.normal_(model.module.classifier[-1].weight, std=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=5e-4, momentum=0.9, nesterov=True)
    class_stats = dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1. / stat) for stat in class_stats])).cuda() * num_class_new
    loss_func = HybridLossWeighted(num_class_new, 512, params.margin, lamda=params.center_lamda, class_stats=class_stats)#WeightedRegularizedTriplet("none")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer,
                                                                      scheduler)
    loss_stats = []
    transforms_augment = nn.Sequential(
        # transforms.Pad([10, 10]),
        # transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(p=1.))
    scripted_transforms_augment = torch.jit.script(transforms_augment).cuda()
    # Additionally train 10 epochs
    for epoch in range(10):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample[:2]
            sample_weights = sample[-1].cuda(non_blocking=True)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            images_augment = scripted_transforms_augment(images)
            label = Variable(label).cuda(non_blocking=True)
            embeddings, outputs = model(images)
            embeddings_augment, _ = model(images_augment)
            loss = loss_func(embeddings, outputs, label, embeddings_augment, sample_weights / batch_size)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            for param in loss_func.center.parameters():
                param.grad.data *= (1. / params.center_lamda)
            optimizer_center.step()
            description = "epoch: {}, Triplet loss: {:.4f}".format(epoch, loss)
            iterator.set_description(description)
        scheduler.step()
    # model.needs_norm = True
    model.eval()
    try:
        to_onnx(model.module,
                torch.randn(2, 3, 256, 128, requires_grad=True, device="cuda"),
                output_names=["embeddings", "outputs"])
    except RuntimeError:
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint.pt")
    return model, loss_stats


def parser():
    def range_type(astr, min=-1., max=6.):
        value = float(astr)
        if min <= value <= max:
            return value
        else:
            raise argparse.ArgumentTypeError('value not in range %s-%s' % (min, max))
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["market1501", "dukemtmc"], default="market1501")
    args.add_argument("--root", type=str, default="~/real-time-ReID-tracking")
    args.add_argument("--ckpt", help="where the checkpoint of vit is, can either be a onnx or pt", type=str,
                      default="vision_transformer_checkpoint.pt")
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="plr_osnet", choices=["seres18",
                                                                            "cares18",
                                                                            "plr_osnet",
                                                                            "vit",
                                                                            "swin_v1",
                                                                            "swin_v2",
                                                                            "baseline"])
    args.add_argument("--epochs", type=int, default=120)
    args.add_argument("--epsilon", help="for polyloss, 0 by default", type=range_type, default=0.0, metavar="[-1, 6]")
    args.add_argument("--margin", help="for triplet loss", default=0.0, type=float)
    args.add_argument("--center_lamda", help="for center loss", default=0.0, type=float)
    args.add_argument("--continual", action="store_true")
    args.add_argument("--accelerate", action="store_true")
    args.add_argument("--renorm", action="store_true")
    args.add_argument("--instance", type=int, default=0)
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    if params.dataset == "market1501":
        dataset = Market1501(root="/".join((params.root, "Market1501")))
    elif params.dataset == "dukemtmc":
        dataset = DukeMTMCreID(root=params.root)
    else:
        raise NotImplementedError("Only market and dukemtmc datasets are supported!\n")
    providers = ["CUDAExecutionProvider"]

    if params.backbone in ("plr_osnet", "seres18", "baseline", "cares18"):
        # No need for cross-domain retrain
        transform_train = transforms.Compose([
            transforms.Resize((256, 128)), # interpolation=3
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            LGT(0.4),
            transforms.RandomGrayscale(0.05),
            # transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(),
        ])
        market_dataset = reidDataset(dataset.train, dataset.num_train_pids, transform_train)
        torch.cuda.empty_cache()
        if params.backbone == "plr_osnet":
            model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_plr_osnet(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                                params.accelerate)
        else:
            if params.backbone == "seres18":
                model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=params.renorm, num_cams=dataset.num_train_cams).cuda()
            elif params.backbone == "cares18":
                model = cares18_ibn(dataset.num_train_pids, renorm=params.renorm, num_cams=dataset.num_train_cams).cuda()
            else:
                model = ft_baseline(dataset.num_train_pids).cuda()
            print("model size: {:.3f} MB".format(check_parameters(model)))
            model = nn.DataParallel(model)
            model, loss_stats = train_cnn(model, market_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                          params.accelerate)

            if params.continual:
                transform_test = transforms.Compose([transforms.Resize((256, 128)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
                merged_datasets = dataset.gallery + dataset.query
                dataset_test = reidDataset(merged_datasets, dataset.num_train_pids, transform_test)

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model.onnx", providers=providers)
                pseudo_labeled_data, num_class_new = produce_pseudo_data(model, dataset_test, dataset.num_gallery_cams, use_onnx=True)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data, num_class_new)
                market_dataset.set_cross_domain()
                model = representation_only(model)
                torch.cuda.empty_cache()
                model, loss_stats = train_cnn_continual(model, market_dataset, num_class_new, params.bs, params.accelerate)
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
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        market_dataset = reidDataset(dataset.train, dataset.num_train_pids, transform_train)

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
                merged_datasets = dataset.gallery + dataset.query
                dataset_test = reidDataset(merged_datasets, dataset.num_train_pids, transform_test)
                # dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4,
                #                               pin_memory=True)

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model.onnx", providers=providers)

                pseudo_labeled_data, num_class_new = produce_pseudo_data(model, dataset_test, dataset.num_gallery_cams, use_onnx=True, use_side=True)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data, num_class_new)
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
                merged_datasets = dataset.gallery + dataset.query
                dataset_test = reidDataset(merged_datasets, dataset.num_train_pids, transform_test)
                # dataloader_test = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4,
                #                               pin_memory=True)

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model.onnx", providers=providers)
                pseudo_labeled_data, num_class_new = produce_pseudo_data(model, dataset_test, dataset.num_gallery_cams, use_onnx=True, use_side=True)
                del dataset_test
                market_dataset.add_pseudo(pseudo_labeled_data, num_class_new)
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
