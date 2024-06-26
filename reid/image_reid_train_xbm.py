import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from backbones.baseline_lite import ft_baseline
from backbones.plr_osnet import plr_osnet
from backbones.SERes18_IBN import seres18_ibn
from backbones.CARes18 import cares18_ibn
from backbones.vision_transformer import vit_t
from backbones.swin_transformer import swin_t
from tricks.XBM import XBM
from train_utils import check_parameters, DataLoaderX, plot_loss, to_numpy
from datasets.dataset_market import Market1501
from datasets.dataset_dukemtmc import DukeMTMCreID
from datasets.dataset_veri776 import VeRi
from train_prepare import WarmupMultiStepLR, WarmUpCosineScheduler, to_onnx
from data_prepare import reidDataset, RandomIdentitySampler_
from data_transforms import get_train_transforms, get_inference_transforms
from inference_utils import diminish_camera_bias
from losses.triplet_losses_xbm import WeightedRegularizedTripletXBM
from losses.hybrid_losses import HybridLoss, HybridLossWeighted
from faiss_utils import compute_jaccard_distance

import argparse
from tqdm import tqdm
import onnxruntime
import madgrad
from sklearn.cluster import DBSCAN
from accelerate import Accelerator

cudnn.deterministic = True
cudnn.benchmark = True


def train_cnn(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    xbm = XBM(4 * batch_size, model.module.classifier[-1].in_features)

    class_stats = dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1. / stat) for stat in class_stats]), dim=-1).cuda() * num_classes
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    loss_func = HybridLoss(num_classes, model.module.classifier[-1].in_features, params.margin, epsilon=params.epsilon, lamda=params.center_lamda,
                           class_stats=class_stats, tao=params.temperature)
    loss_func_xbm = WeightedRegularizedTripletXBM()
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)

    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(dataset, batch_size, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    lr_scheduler = WarmUpCosineScheduler(optimizer, epochs)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=not params.instance,
                             pin_memory=True, sampler=custom_sampler, drop_last=False)
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
            images, label, cams = sample[:3]
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            # images_flip = scripted_transforms_augment(images)
            label = Variable(label).cuda(non_blocking=True)
            # cams = cams.cuda(non_blocking=True)
            embeddings, outputs = model(images)  # , cams)
            loss = loss_func(embeddings, outputs, label)
            # loss = loss_func(embeddings, outputs, label, embeddings_augment)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)

            if epoch > 25:
                xbm.enqueue_dequeue(embeddings.detach(), label.detach())
                xbm_feats, xbm_targets = xbm.get()
                xbm_loss = loss_func_xbm(embeddings, label, xbm_feats, xbm_targets)
                loss = loss + xbm_loss

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
    dummy_input = torch.randn(2, 3, 224, int(ratio * 224), requires_grad=True, device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 256, 128, requires_grad=True, device="cuda")
    try:
        to_onnx(model.module,
                dummy_input,
                # torch.ones(1, dtype=torch.long)),
                params.dataset,
                # input_names=["input", "index"],
                output_names=["embeddings", "outputs"])
    except:  # There may be op issue
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint_{}.pt".format(params.dataset))
    return model, loss_stats


def train_cnn_sie(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    xbm = XBM(4 * batch_size, model.module.classifier[-1].in_features)

    class_stats = dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1. / stat) for stat in class_stats]), dim=-1).cuda() * num_classes
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    loss_func = HybridLoss(num_classes, model.module.classifier[-1].in_features, params.margin, epsilon=params.epsilon, lamda=params.center_lamda,
                           class_stats=class_stats, tao=params.temperature)
    loss_func_xbm = WeightedRegularizedTripletXBM()
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)

    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(dataset, batch_size, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    lr_scheduler = WarmUpCosineScheduler(optimizer, epochs)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=not params.instance,
                             pin_memory=True, sampler=custom_sampler, drop_last=False)
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
            images, label, cams, seqs = sample
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            cams = cams.cuda(non_blocking=True)
            embeddings, outputs = model(images, cams)
            loss = loss_func(embeddings, outputs, label)
            # loss = loss_func(embeddings, outputs, label, embeddings_augment)
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)

            if epoch > 10:
                xbm.enqueue_dequeue(embeddings.detach(), label.detach())
                xbm_feats, xbm_targets = xbm.get()
                xbm_loss = loss_func_xbm(embeddings, label, xbm_feats, xbm_targets)
                loss = loss + xbm_loss

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
    dummy_input = torch.randn(2, 3, 224, int(ratio * 224), requires_grad=True, device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 256, 128, requires_grad=True, device="cuda")
    try:
        to_onnx(model.module,
                (dummy_input,
                 torch.ones(1, dtype=torch.long)),
                params.dataset + "_sie",
                input_names=["input", "index"],
                output_names=["embeddings", "outputs"])
    except:  # There may be op issue
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint_{}_sie.pt".format(params.dataset))
    return model, loss_stats


def train_plr_osnet(model, dataset, batch_size=8, epochs=25, num_classes=517, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)
    model.train()
    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(dataset, batch_size, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = madgrad.MADGRAD(model.parameters(), lr=0.01, weight_decay=5e-4)
    lr_scheduler = WarmUpCosineScheduler(optimizer, epochs)
    loss_func1 = HybridLoss(num_classes, model.module.classifier1.in_features, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    loss_func2 = HybridLoss(num_classes, model.module.classifier2.in_features, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=not params.instance,
                             pin_memory=True, sampler=custom_sampler)
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
    dummy_input = torch.randn(2, 3, 224, int(ratio * 224), requires_grad=True,
                              device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 256, 128,
                                                                                          requires_grad=True,
                                                                                          device="cuda")
    to_onnx(model.module,
            dummy_input,
            params.dataset,
            output_names=["y1", "y2", "fea"])
    torch.save(model.state_dict(), "checkpoint/plr_osnet_checkpoint_{}.pt".format(params.dataset))
    return model, loss_stats


def train_transformer_model(model, dataset, feat_dim=384, batch_size=8, epochs=25, num_classes=517,
                            all_cams=6, all_seqs=6, accelerate=False):
    if params.ckpt and os.path.exists(params.ckpt):
        model.eval()
        model_state_dict = torch.load(params.ckpt)
        model.load_state_dict(model_state_dict, strict=False)

    model.train()
    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(dataset, batch_size, params.instance)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.008, weight_decay=1e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    lr_scheduler = WarmUpCosineScheduler(optimizer, epochs)
    loss_func = HybridLoss(num_classes, feat_dim, params.margin, epsilon=params.epsilon, lamda=params.center_lamda)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=not params.instance,
                             pin_memory=True, sampler=custom_sampler)
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
            images, label, cam, seqid = sample
            assert all(seqid) < all_seqs and all(cam) < all_cams
            if cam is not None and any(cam) >= 0 and seqid is not None and any(seqid) >= 0:
                view_index = cam + all_cams * seqid
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
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True,
                              device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 448, 224,
                                                                                          requires_grad=True,
                                                                                          device="cuda")
    to_onnx(model.module,
            (dummy_input,
             torch.ones(1, dtype=torch.long)),
            params.dataset,
            input_names=["input", "index"],
            output_names=["embeddings", "outputs"])
    torch.save(model.state_dict(), "checkpoint/vision_transformer_checkpoint_{}.pt".format(params.dataset))
    return model, loss_stats


def side_info_only(model):
    model.train()
    for name, params in model.named_parameters():
        if name.endswith("side_info_embedding"):
            params.requires_grad = True
        else:
            params.requires_grad = False
    return model


def produce_pseudo_data(model,
                        dataset_test,
                        merged_datasets,
                        all_cam=6,
                        use_onnx=False,
                        use_side=False) -> tuple:
    model.eval()
    pseudo_data = []
    embeddings = []
    cams = []
    seqs = []
    if not use_onnx:
        dataloader = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for iteration, sample in enumerate(dataloader, 0):
                img, _, cam, seq = sample
                img = img.cuda(non_blocking=True)
                if use_side:
                    embedding, output = model(img, cam)  # cam + all_cam * seq
                else:
                    embedding, output = model(img)
                embeddings.append(torch.from_numpy(np.concatenate((embedding, output), axis=1)))
                cams.append(cam)
                seqs.append(seq)
    else:
        # experiment
        dataloader = DataLoaderX(dataset_test, batch_size=params.bs, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for iteration, sample in enumerate(dataloader, 0):
                img, _, cam, seq = sample
                if use_side:
                    ort_inputs = {'input': to_numpy(img),
                                  "index": to_numpy(cam)}  # cam + all_cam * seq
                else:
                    ort_inputs = {'input': to_numpy(img)}
                embedding, output = ort_session.run(["embeddings", "outputs"], ort_inputs)
                # embeddings.append(torch.from_numpy(np.concatenate((embedding, output), axis=1)))
                embeddings.append(torch.from_numpy(embedding))
                cams.append(cam)
                seqs.append(seq)
    embeddings = F.normalize(torch.cat(embeddings, dim=0), dim=1, p=2)
    cams = torch.cat(cams, dim=0)
    seqs = torch.cat(seqs, dim=0)
    embeddings = diminish_camera_bias(embeddings, cams)
    # dists = euclidean_dist(embeddings, embeddings)
    dists = compute_jaccard_distance(embeddings, search_option=0)
    cluster_method = DBSCAN(eps=params.eps, min_samples=min(10, all_cam + 1), metric="precomputed", n_jobs=-1)
    labels = cluster_method.fit_predict(dists)
    centroids = torch.zeros((max(labels)+1, embeddings.size(1)))
    for i, label in enumerate(labels):
        if label != -1:
            pseudo_data.append((
                merged_datasets[i][0],
                label + dataset.num_train_pids,
                cams[i].item(),
                seqs[i].item()
            ))
    for i in range(max(labels)+1):
        centroids[i] = embeddings[labels == i].mean(dim=0)
    print("Inference completed! {} more pseudo-labels obtained!".format(len(pseudo_data)))
    return pseudo_data, max(labels) + 1 + dataset.num_train_pids, centroids


def train_cnn_continual(model, merged_dataset, num_class_new, centroids, batch_size=8, accelerate=False):
    model.train()
    prev_weights = model.module.classifier[-1].weight.data
    tmp_feat_dim = model.module.classifier[-1].in_features
    model.module.classifier[-1] = nn.Linear(tmp_feat_dim, num_class_new, bias=False, device="cuda")
    # nn.init.normal_(model.module.classifier[-1].weight, std=0.001)
    model.module.classifier[-1].weight.data[:dataset.num_train_pids] = prev_weights
    model.module.classifier[-1].weight.data[dataset.num_train_pids:] = centroids
    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(merged_dataset, batch_size, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=7e-5, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=5e-4, momentum=0.9, nesterov=True)
    class_stats = merged_dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1. / stat) for stat in class_stats]), dim=-1).cuda() * num_class_new
    loss_func = HybridLossWeighted(num_class_new, tmp_feat_dim, params.margin, lamda=params.center_lamda,
                                   class_stats=class_stats, centroids=centroids, tao=2.)  # WeightedRegularizedTriplet("none")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)
    dataloader = DataLoaderX(merged_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=not params.instance,
                             pin_memory=True,
                             sampler=custom_sampler)
    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer,
                                                                      scheduler)
    loss_stats = []

    # Additionally train 40 epochs
    for epoch in range(40):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample[:2]
            sample_weights = sample[-1].cuda(non_blocking=True)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            # images_augment = scripted_transforms_augment(images)
            label = Variable(label).cuda(non_blocking=True)
            embeddings, outputs = model(images)
            # embeddings_augment, _ = model(images_augment)
            loss = loss_func(embeddings, outputs, label, weights=sample_weights / batch_size)
            # loss = loss_func(embeddings, outputs, label, embeddings_augment, sample_weights / batch_size)
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
    model.eval()
    dummy_input = torch.randn(2, 3, 224, int(ratio * 224), requires_grad=True,
                              device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 256, 128,
                                                                                          requires_grad=True,
                                                                                          device="cuda")
    try:
        to_onnx(model.module,
                dummy_input,
                params.dataset + "_continual",
                output_names=["embeddings", "outputs"])
    except RuntimeError:
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint_{}_continual.pt".format(params.dataset))
    return model, loss_stats


def train_cnn_continual_sie(model, merged_dataset, num_class_new, centroids, batch_size=8, accelerate=False):
    model.train()
    prev_weights = model.module.classifier[-1].weight.data
    tmp_feat_dim = model.module.classifier[-1].in_features
    model.module.classifier[-1] = nn.Linear(tmp_feat_dim, num_class_new, bias=False, device="cuda")
    # nn.init.normal_(model.module.classifier[-1].weight, std=0.001)
    model.module.classifier[-1].weight.data[:dataset.num_train_pids] = prev_weights
    model.module.classifier[-1].weight.data[dataset.num_train_pids:] = centroids
    if params.instance > 0:
        custom_sampler = RandomIdentitySampler_(merged_dataset, batch_size, params.instance)
        optimizer = torch.optim.Adam(model.parameters(), lr=7e-5, weight_decay=5e-4)
    else:
        custom_sampler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=5e-4, momentum=0.9, nesterov=True)
    class_stats = merged_dataset.get_class_stats()
    class_stats = F.softmax(torch.stack([torch.tensor(1. / stat) for stat in class_stats]), dim=-1).cuda() * num_class_new
    loss_func = HybridLossWeighted(num_class_new, tmp_feat_dim, params.margin, lamda=params.center_lamda,
                                   class_stats=class_stats, centroids=centroids, tao=2.)  # WeightedRegularizedTriplet("none")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    optimizer_center = torch.optim.SGD(loss_func.center.parameters(), lr=0.5)
    dataloader = DataLoaderX(merged_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=not params.instance,
                             pin_memory=True,
                             sampler=custom_sampler)
    if accelerate:
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer,
                                                                      scheduler)
    loss_stats = []

    # Additionally train 40 epochs
    for epoch in range(40):
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label, cams = sample[:3]
            sample_weights = sample[-1].cuda(non_blocking=True)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            embeddings, outputs = model(images, cams)
            loss = loss_func(embeddings, outputs, label, weights=sample_weights / batch_size)
            # loss = loss_func(embeddings, outputs, label, embeddings_augment, sample_weights / batch_size)
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
    model.eval()
    dummy_input = torch.randn(2, 3, 224, int(ratio * 224), requires_grad=True,
                              device="cuda") if params.dataset == "veri" else torch.randn(2, 3, 256, 128,
                                                                                          requires_grad=True,
                                                                                          device="cuda")
    try:
        to_onnx(model.module,
                (dummy_input,
                 torch.ones(1, dtype=torch.long)),
                params.dataset + "_sie_continual",
                input_names=["input", "index"],
                output_names=["embeddings", "outputs"])
    except RuntimeError:
        pass
    torch.save(model.state_dict(), "checkpoint/cnn_net_checkpoint_{}_sie_continual.pt".format(params.dataset))
    return model, loss_stats


def parser():
    def range_type(astr, min=-1., max=6.):
        value = float(astr)
        if min <= value <= max:
            return value
        else:
            raise argparse.ArgumentTypeError('value not in range %s-%s' % (min, max))

    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["market1501", "dukemtmc", "veri"], default="market1501")
    args.add_argument("--root", type=str, default="~/real-time-ReID-tracking")
    args.add_argument("--ckpt", help="where the checkpoint of vit is, can either be a onnx or pt", type=str,
                      default="vision_transformer_checkpoint.pt")
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="seres18", choices=["seres18",
                                                                          "cares18",
                                                                          "plr_osnet",
                                                                          "vit",
                                                                          "swin_v1",
                                                                          "swin_v2",
                                                                          "baseline"])
    args.add_argument("--epochs", type=int, default=160)
    args.add_argument("--epsilon", help="for polyloss, 0 by default", type=range_type, default=0.0, metavar="[-1, 6]")
    args.add_argument("--margin", help="for triplet loss", default=0.0, type=float)
    args.add_argument("--center_lamda", help="for center loss", default=0.0005, type=float)
    args.add_argument("--eps", default=0.5, type=float, help="clustering eps for continual training")
    args.add_argument("--continual", action="store_true", help="continual training")
    args.add_argument("--accelerate", action="store_true")
    args.add_argument("--renorm", action="store_true")
    args.add_argument("--instance", type=int, default=0)
    args.add_argument("--sie", action="store_true", help="side information embedding")
    args.add_argument("--temperature", default=1, type=int)
    args.add_argument("--cam_factor", default=0., type=float)
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    if params.dataset == "market1501":
        dataset = Market1501(root="/".join((params.root, "Market1501")))
        ratio = 0.5
    elif params.dataset == "dukemtmc":
        dataset = DukeMTMCreID(root=params.root)
        ratio = 0.5
    elif params.dataset == "veri":
        dataset = VeRi(root=params.root)
        ratio = dataset.get_ratio()
    else:
        raise NotImplementedError("Only market, dukemtmc and veri datasets are supported!\n")
    providers = ["CUDAExecutionProvider"]

    if params.backbone in ("plr_osnet", "seres18", "baseline", "cares18"):
        # No need for cross-domain retrain
        transform_train = get_train_transforms(params.dataset, ratio)
        source_dataset = reidDataset(dataset.train, dataset.num_train_pids, transform_train)
        torch.cuda.empty_cache()
        if params.backbone == "plr_osnet":
            model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_plr_osnet(model, source_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                                params.accelerate)
        else:
            if params.backbone == "seres18":
                model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=params.renorm,
                                    num_cams=dataset.num_train_cams, cam_factor=params.cam_factor).cuda()
            elif params.backbone == "cares18":
                model = cares18_ibn(dataset.num_train_pids, renorm=params.renorm, num_cams=dataset.num_train_cams,
                                    non_iid=params.instance, cam_factor=params.cam_factor).cuda()
            else:
                model = ft_baseline(dataset.num_train_pids).cuda()
            print("model size: {:.3f} MB".format(check_parameters(model)))
            model = nn.DataParallel(model)
            if params.sie:
                model, loss_stats = train_cnn_sie(model, source_dataset, params.bs, params.epochs,
                                                  dataset.num_train_pids,
                                                  params.accelerate)
            else:
                model, loss_stats = train_cnn(model, source_dataset, params.bs, params.epochs, dataset.num_train_pids,
                                              params.accelerate)

            if params.continual:
                transform_test = get_inference_transforms(params.dataset, ratio)
                merged_datasets = dataset.gallery + dataset.query
                dataset_test = reidDataset(merged_datasets, dataset.num_train_pids, transform_test)

                ort_session = onnxruntime.InferenceSession("checkpoint/reid_model_{}_sie.onnx".format(params.dataset) if params.sie else "checkpoint/reid_model_{}.onnx".format(params.dataset),
                                                           providers=providers)
                pseudo_labeled_data, num_class_new, centroids = produce_pseudo_data(model, dataset_test, merged_datasets,
                                                                         dataset.num_gallery_cams, use_onnx=True,
                                                                         use_side=params.sie)
                del dataset_test
                source_dataset.add_pseudo(pseudo_labeled_data, num_class_new)
                source_dataset.set_cross_domain()
                # model = representation_only(model)
                torch.cuda.empty_cache()
                if params.sie:
                    model, loss_stats = train_cnn_continual_sie(model, source_dataset, num_class_new, centroids, params.bs,
                                                                params.accelerate)
                else:
                    model, loss_stats = train_cnn_continual(model, source_dataset, num_class_new, centroids, params.bs,
                                                            params.accelerate)
                source_dataset.reset_cross_domain()

    else:
        transform_train = get_train_transforms(params.dataset, transformer_model=True)
        transform_test = get_inference_transforms(params.dataset, transformer_model=True)
        source_dataset = reidDataset(dataset.train, dataset.num_train_pids, transform_train)

        if params.backbone.startswith("vit"):
            model = vit_t(img_size=(448, 224) if params.dataset != "veri" else (224, 224), num_classes=dataset.num_train_pids, loss="triplet",
                          camera=dataset.num_train_cams,
                          sequence=dataset.num_train_seqs, side_info=True).cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_transformer_model(model, source_dataset, 384,
                                                        params.bs, params.epochs,
                                                        dataset.num_train_pids,
                                                        dataset.num_train_cams,
                                                        dataset.num_train_seqs,
                                                        params.accelerate)
        elif params.backbone.startswith("swin"):
            model = swin_t(num_classes=dataset.num_train_pids, loss="triplet",
                           camera=dataset.num_train_cams, sequence=dataset.num_train_seqs,
                           side_info=True,
                           version=params.backbone[-2:]).cuda()
            model = nn.DataParallel(model)
            model, loss_stats = train_transformer_model(model, source_dataset, 96,
                                                        params.bs, params.epochs,
                                                        dataset.num_train_pids,
                                                        dataset.num_train_cams,
                                                        dataset.num_train_seqs,
                                                        params.accelerate)

        else:
            raise NotImplementedError

    plot_loss(loss_stats)
