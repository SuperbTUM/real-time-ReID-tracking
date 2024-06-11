import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
import numpy as np
from tqdm import tqdm

from backbones.baseline_lite import ft_baseline
from backbones.SERes18_IBN import seres18_ibn
from backbones.CARes18 import cares18_ibn
from backbones.plr_osnet import plr_osnet
from backbones.vision_transformer import vit_t
from backbones.swin_transformer import swin_t
from backbones.resnet50 import ft_net

from datasets.dataset_market import Market1501
from datasets.dataset_dukemtmc import DukeMTMCreID
from datasets.dataset_veri776 import VeRi
from train_utils import to_numpy, DataLoaderX
from data_prepare import reidDataset
from data_transforms import get_inference_transforms, get_inference_transforms_flipped
from inference_utils import diminish_camera_bias
from faiss_utils import compute_jaccard_distance
from evaluate import evaluate_all


def inference(model, dataloader, all_cam=6, use_onnx=True, use_side=False):
    torch.cuda.empty_cache()
    model.eval()
    embeddings_total = []
    true_labels = []
    true_cams = []
    if not use_onnx:
        with torch.no_grad():
            for sample in tqdm(dataloader):
                img, true_label, cam, seq = sample
                img = img.cuda(non_blocking=True)
                if use_side:
                    side_index = (cam + all_cam * seq).cuda(non_blocking=True)
                    embeddings, _ = model(img, side_index)
                else:
                    embeddings, _ = model(img)
                embeddings_total.append(embeddings)
                true_labels.append(true_label)
                true_cams.append(cam)
    else:
        for sample in tqdm(dataloader):
            img, true_label, cam, seq = sample
            if not use_side:
                ort_inputs = {'input': to_numpy(img)}
                # input_ortvalue.update_inplace(to_numpy(img))
            else:
                ort_inputs = {'input': to_numpy(img),
                              "index": to_numpy(cam + all_cam * seq)}
                # input_ortvalue.update_inplace(to_numpy(img))
                # index_ortvalue.update_inplace(to_numpy(cam + all_cam * seq))
            # ort_session.run_with_iobinding(io_binding)
            embeddings = ort_session.run(["embeddings", "outputs"], ort_inputs)[
                0]  # embeddings_ortvalue.numpy()#ort_session.run(["embeddings", "outputs"], ort_inputs)[0]
            embeddings = torch.from_numpy(embeddings)
            embeddings_total.append(embeddings)
            true_labels.append(true_label)
            true_cams.append(cam)
    embed_dim = embeddings_total[-1].size(1)
    try:
        embeddings_total = torch.stack(embeddings_total).view(-1, embed_dim)
        true_labels = torch.stack(true_labels).flatten()
        true_cams = torch.stack(true_cams).flatten()
    except RuntimeError:
        embeddings_total = torch.cat((torch.stack(embeddings_total[:-1]).view(-1, embed_dim), embeddings_total[-1]),
                                     dim=0)
        true_labels = torch.cat((torch.stack(true_labels[:-1]).flatten(), true_labels[-1]), dim=0)
        true_cams = torch.cat((torch.stack(true_cams[:-1]).flatten(), true_cams[-1]), dim=0)
    return embeddings_total, true_labels, true_cams


def inference_efficient(model, dataloader1, dataloader2, use_side=False, use_onnx=False):
    torch.cuda.empty_cache()
    model.eval()
    embeddings_total1 = []
    embeddings_total2 = []
    true_labels = []
    true_cams = []
    true_seqs = []
    if use_onnx:
        for sample1, sample2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
            assert len(sample1) == len(sample2)
            img1, true_label, cam, seq = sample1
            img2 = sample2[0]
            img = torch.cat((img1, img2), dim=0)
            if not use_side:
                ort_inputs = {'input': to_numpy(img)}
            else:
                ort_inputs = {'input': to_numpy(img),
                              "index": np.tile(to_numpy(cam), 2)}
            # experimental
            embeddings, outputs = ort_session.run(["embeddings", "outputs"], ort_inputs)
            if params.cross_domain:
                embeddings = torch.from_numpy(embeddings)  # pending???
            else:
                embeddings = torch.cat(
                    (F.normalize(torch.from_numpy(embeddings), dim=1), F.normalize(torch.from_numpy(outputs), dim=1)),
                    dim=1)
            embeddings_total1.append(embeddings[:(len(embeddings) >> 1)])
            embeddings_total2.append(embeddings[(len(embeddings) >> 1):])
            true_labels.append(true_label)
            true_cams.append(cam)
            true_seqs.append(seq)
    else:
        with torch.no_grad():
            for sample1, sample2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
                assert len(sample1) == len(sample2)
                img1, true_label, cam, seq = sample1
                img2 = sample2[0]
                img = torch.cat((img1, img2), dim=0).cuda()
                if not use_side:
                    embeddings, outputs = model(img)
                else:
                    embeddings, outputs = model(img, cam.repeat(2))
                embeddings = embeddings.cpu()
                outputs = outputs.cpu()
                embeddings = torch.cat((F.normalize(embeddings, dim=1), F.normalize(outputs, dim=1)), dim=1)
                embeddings_total1.append(embeddings[:(len(embeddings) >> 1)])
                embeddings_total2.append(embeddings[(len(embeddings) >> 1):])
                true_labels.append(true_label)
                true_cams.append(cam)
                true_seqs.append(seq)

    embeddings_total1 = torch.cat(embeddings_total1, dim=0)
    embeddings_total2 = torch.cat(embeddings_total2, dim=0)
    true_labels = torch.cat(true_labels, dim=0).flatten()
    true_cams = torch.cat(true_cams, dim=0).flatten()
    true_seqs = torch.cat(true_seqs, dim=0).flatten()
    return (embeddings_total1, embeddings_total2), true_labels, true_cams, true_seqs


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["market1501", "dukemtmc", "veri"], default="market1501")
    args.add_argument("--root", type=str, default="~/real-time-ReID-tracking")
    args.add_argument("--ckpt", help="where the checkpoint of vit is, can either be a onnx or pt", type=str,
                      default="checkpoint/vision_transformer_checkpoint.pt")
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--backbone", type=str, default="plr_osnet", choices=["seres18",
                                                                            "cares18",
                                                                            "plr_osnet",
                                                                            "vit",
                                                                            "swin_v1",
                                                                            "swin_v2",
                                                                            "resnet50",
                                                                            "baseline"])
    args.add_argument("--sie", action="store_true")
    args.add_argument("--renorm", action="store_true")
    args.add_argument("--eps", type=float, default=0.5)
    args.add_argument("--cross_domain", action="store_true")
    args.add_argument("--market_attribute_path", default="Market-1501_Attribute/market_attribute.mat", type=str)
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

    if params.backbone == "plr_osnet":
        transform_test = get_inference_transforms(params.dataset, ratio)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, ratio)
        model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
    elif params.backbone in ("seres18", "cares18"):
        transform_test = get_inference_transforms(params.dataset, ratio)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, ratio, strong_inference=True)
        if params.backbone == "seres18":
            model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=params.renorm,
                                num_cams=dataset.num_train_cams).cuda()
        else:
            model = cares18_ibn(dataset.num_train_pids, renorm=params.renorm, num_cams=dataset.num_train_cams).cuda()
    elif params.backbone == "resnet50":
        transform_test = get_inference_transforms(params.dataset, ratio)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, ratio)
        model = ft_net(dataset.num_train_pids).cuda()
    elif params.backbone == "baseline":
        transform_test = get_inference_transforms(params.dataset, ratio)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, ratio)
        model = ft_baseline(dataset.num_train_pids).cuda()
    elif params.backbone == "vit":
        transform_test = get_inference_transforms(params.dataset, transformer_model=True)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, transformer_model=True)
        model = vit_t(img_size=(448, 224) if params.dataset in ("market1501", "dukemtmc") else (224, 224),
                      num_classes=dataset.num_train_pids, loss="triplet",
                      camera=dataset.num_train_cams,
                      sequence=dataset.num_train_seqs, side_info=True).cuda()
    elif params.backbone.startswith("swin"):
        transform_test = get_inference_transforms(params.dataset, transformer_model=True)
        transform_test_flip = get_inference_transforms_flipped(params.dataset, transformer_model=True)
        model = swin_t(num_classes=dataset.num_train_pids, loss="triplet",
                       camera=dataset.num_train_cams, sequence=dataset.num_train_seqs,
                       side_info=True,
                       version=params.backbone[-2:]).cuda()
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model.eval()
    if params.ckpt.endswith("pt") or params.ckpt.endswith("pth"):
        model_weights = torch.load(params.ckpt)
        if not params.sie:
            try:
                del model_weights["module.cam_bias"]
            except KeyError:
                pass
        last_layer_weights = list(model_weights.values())[-1].size(0)
        last_layer_model = list(model.module.named_children())[-1][1]
        if isinstance(last_layer_model, nn.Sequential):
            last_layer_out_feat = last_layer_model[-1].out_features
        else:
            last_layer_out_feat = last_layer_model.out_features
        if last_layer_out_feat != last_layer_weights:
            last_layer_name = list(model.module.named_children())[-1][0]
            if isinstance(last_layer_model, nn.Sequential):
                last_layer_in_feat = last_layer_model[-1].in_features
            else:
                last_layer_in_feat = last_layer_model.in_features
            setattr(model, "module." + last_layer_name + (".0" if isinstance(last_layer_model, nn.Sequential) else ""), nn.Linear(last_layer_in_feat, last_layer_weights, bias=False).cuda())
        model.load_state_dict(model_weights, strict=False)
        use_onnx = False

    else:
        # providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(params.ckpt, providers=providers)
        use_onnx = True
    # experimental
    reid_gallery = reidDataset(dataset.gallery, dataset.num_train_pids, transform_test, False)
    dataloader1 = DataLoaderX(reid_gallery, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    reid_gallery.transform = transform_test_flip
    dataloader2 = DataLoaderX(reid_gallery, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    gallery_embeddings, gallery_labels, gallery_cams, gallery_seqs = inference_efficient(model, dataloader1,
                                                                                         dataloader2, params.sie,
                                                                                         use_onnx)
    gallery_embeddings1 = gallery_embeddings[0]
    gallery_embeddings2 = gallery_embeddings[1]

    gallery_embeddings = (gallery_embeddings1 + gallery_embeddings2) / 2.0
    gallery_embeddings = F.normalize(gallery_embeddings, dim=1)

    reid_query = reidDataset(dataset.query, dataset.num_train_pids, transform_test, False)
    dataloader1 = DataLoaderX(reid_query, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    reid_query.transform = transform_test_flip
    dataloader2 = DataLoaderX(reid_query, batch_size=params.bs, num_workers=4, shuffle=False,
                              pin_memory=True)
    query_embeddings, query_labels, query_cams, query_seqs = inference_efficient(model, dataloader1, dataloader2,
                                                                                 params.sie, use_onnx)
    query_embeddings1 = query_embeddings[0]
    query_embeddings2 = query_embeddings[1]
    # query_embeddings1 = F.normalize(query_embeddings1, dim=1)
    # query_embeddings2 = F.normalize(query_embeddings2, dim=1)

    query_embeddings = (query_embeddings1 + query_embeddings2) / 2.0
    query_embeddings = F.normalize(query_embeddings, dim=1)

    merged_embeddings = torch.cat((gallery_embeddings, query_embeddings), dim=0)
    merged_cams = torch.cat((gallery_cams, query_cams), dim=0)
    merged_seqs = torch.cat((gallery_seqs, query_seqs), dim=0)

    merged_embeddings = diminish_camera_bias(merged_embeddings, merged_cams)

    if params.dataset == "market1501":
        from tricks.additional_market_attributes import get_attribute_dist

        attribute_dist = get_attribute_dist(
            [label for _, label, _, _ in reid_gallery.images] + [label for _, label, _, _ in reid_query.images],
            params.market_attribute_path)

    # from losses.utils import euclidean_dist
    dists = compute_jaccard_distance(merged_embeddings,
                                     search_option=0)  # euclidean_dist(merged_embeddings, merged_embeddings) #
    dists[dists < 0] = 0.
    # dists[dists > 1] = 1.
    if params.dataset == "market1501":
        dists += attribute_dist
    try:
        from cuml import DBSCAN

        print("CUML Imported!")
        cluster_method = DBSCAN(eps=params.eps, min_samples=min(10, dataset.num_gallery_cams + 1), metric="precomputed")
        dists = dists.cpu().numpy()
    except ImportError:
        from sklearn.cluster import DBSCAN

        cluster_method = DBSCAN(eps=params.eps, min_samples=min(10, dataset.num_gallery_cams + 1), metric="precomputed",
                                n_jobs=-1)
    pseudo_labels = cluster_method.fit_predict(dists)
    indices_pseudo = (pseudo_labels != -1)
    num_labels = max(pseudo_labels) + 1
    assert num_labels >= 0.2 * dataset.num_train_pids
    # merged_seqs = merged_seqs * dataset.num_gallery_cams * num_labels + merged_cams * num_labels + pseudo_labels
    merged_seqs = merged_seqs * num_labels + pseudo_labels

    from inference_utils import smooth_tracklets

    merged_embeddings = smooth_tracklets(merged_embeddings, merged_seqs, indices_pseudo)

    gallery_embeddings = merged_embeddings[:len(dataset.gallery)]
    query_embeddings = merged_embeddings[len(dataset.gallery):]

    CMC, ap = evaluate_all(query_embeddings,
                           query_labels,
                           query_cams,
                           gallery_embeddings,
                           gallery_labels,
                           gallery_cams)
