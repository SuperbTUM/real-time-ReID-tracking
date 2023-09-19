import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from backbones.baseline_lite import ft_baseline
from backbones.SERes18_IBN import seres18_ibn
from backbones.CARes18 import cares18_ibn
from backbones.plr_osnet import plr_osnet
from backbones.vision_transformer import vit_t
from backbones.swin_transformer import swin_t
from backbones.resnet50 import ft_net

from dataset_market import Market1501
from dataset_dukemtmc import DukeMTMCreID
from dataset_veri776 import VeRi
from train_utils import to_numpy, DataLoaderX
from data_prepare import reidDataset
from inference_utils import diminish_camera_bias
from faiss_utils import compute_jaccard_distance

# @credit to Zhedong
#######################################################################
# Evaluate
def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def inference(model, dataloader, all_cam=6, use_onnx=True, use_side=False):
    torch.cuda.empty_cache()
    model.eval()
    embeddings_total = []
    true_labels = []
    true_cams = []
    if not use_onnx:
        with torch.no_grad():
            for sample in tqdm(dataloader):
                if len(sample) == 2:
                    img, true_label = sample
                    cam = seq = None
                elif len(sample) == 3:
                    img, true_label, cam = sample
                    seq = 0
                else:
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
        # experimental
        # input_example = np.random.rand(1, 3, 256, 128).astype(np.float32)  # (1, 3, 448, 224)
        # index_example = np.zeros((1, ), dtype=int)
        # embeddings_example = np.random.rand(1, 512).astype(np.float32) # seres18
        # outputs_example = np.random.rand(1, 751).astype(np.float32) # market1501
        # input_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_example, 'cuda', 0)
        # index_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(index_example, 'cuda', 0)
        # embeddings_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(embeddings_example, 'cuda', 0)
        # outputs_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(outputs_example, 'cuda', 0)
        # io_binding = ort_session.io_binding()
        # io_binding.bind_ortvalue_input('input', input_ortvalue)
        # if use_side:
        #     io_binding.bind_ortvalue_input('index', index_ortvalue)
        # io_binding.bind_ortvalue_output('embeddings', embeddings_ortvalue)
        # io_binding.bind_ortvalue_output('outputs', outputs_ortvalue)
        # ort_session.run_with_iobinding(io_binding)
        for sample in tqdm(dataloader):
            if len(sample) == 2:
                img, true_label = sample
                cam = seq = None
            elif len(sample) == 3:
                img, true_label, cam = sample
                seq = 0
            else:
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
            embeddings = ort_session.run(["embeddings", "outputs"], ort_inputs)[0]#embeddings_ortvalue.numpy()#ort_session.run(["embeddings", "outputs"], ort_inputs)[0]
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
        embeddings_total = torch.cat((torch.stack(embeddings_total[:-1]).view(-1, embed_dim), embeddings_total[-1]), dim=0)
        true_labels = torch.cat((torch.stack(true_labels[:-1]).flatten(), true_labels[-1]), dim=0)
        true_cams = torch.cat((torch.stack(true_cams[:-1]).flatten(), true_cams[-1]), dim=0)
    return embeddings_total, true_labels, true_cams


def inference_efficient(model, dataloader1, dataloader2, all_cam=6, use_side=False):
    """Only support onnx inference"""
    torch.cuda.empty_cache()
    model.eval()
    embeddings_total1 = []
    embeddings_total2 = []
    true_labels = []
    true_cams = []
    true_seqs = []
    for sample1, sample2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        assert len(sample1) == len(sample2)
        if len(sample1) == 2:
            img1, true_label = sample1
            cam = seq = None
        elif len(sample1) == 3:
            img1, true_label, cam = sample1
            seq = torch.zeros(img1.size(0), dtype=torch.int32)
        else:
            img1, true_label, cam, seq = sample1
        img2 = sample2[0]
        img = torch.cat((img1, img2), dim=0)
        if not use_side:
            ort_inputs = {'input': to_numpy(img)}
            # input_ortvalue.update_inplace(to_numpy(img))
        else:
            ort_inputs = {'input': to_numpy(img),
                          "index": np.repeat(to_numpy(cam), 2)}
            # input_ortvalue.update_inplace(to_numpy(img))
            # index_ortvalue.update_inplace(to_numpy(cam + all_cam * seq))
        # ort_session.run_with_iobinding(io_binding)
        # experimental
        embeddings, outputs = ort_session.run(["embeddings", "outputs"], ort_inputs)  # embeddings_ortvalue.numpy()#ort_session.run(["embeddings", "outputs"], ort_inputs)[0]
        embeddings = torch.from_numpy(np.concatenate((embeddings, outputs), axis=1))
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
    args.add_argument("--eps", type=float, default=0.25)
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    if params.dataset == "market1501":
        dataset = Market1501(root="/".join((params.root, "Market1501")))
    elif params.dataset == "dukemtmc":
        dataset = DukeMTMCreID(root=params.root)
    elif params.dataset == "veri":
        dataset = VeRi(root=params.root)
    else:
        raise NotImplementedError("Only market, dukemtmc and veri datasets are supported!\n")

    if params.backbone == "plr_osnet":
        transform_test = transforms.Compose([transforms.Resize((256, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        model = plr_osnet(num_classes=dataset.num_train_pids, loss='triplet').cuda()
    elif params.backbone in ("seres18", "cares18"):
        transform_test = transforms.Compose([transforms.Resize((256, 128)), # interpolation=3
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)), # experimental
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        if params.backbone == "seres18":
            model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=params.renorm).cuda()
        else:
            model = cares18_ibn(dataset.num_train_pids, renorm=params.renorm).cuda()
    elif params.backbone == "resnet50":
        transform_test = transforms.Compose([transforms.Resize((256, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        model = ft_net(dataset.num_train_pids).cuda()
    elif params.backbone == "baseline":
        transform_test = transforms.Compose([transforms.Resize((256, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        model = ft_baseline(dataset.num_train_pids).cuda()
    elif params.backbone == "vit":
        transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((448, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        model = vit_t(img_size=(448, 224), num_classes=dataset.num_train_pids, loss="triplet",
                      camera=dataset.num_train_cams,
                      sequence=dataset.num_train_seqs, side_info=True).cuda()
    elif params.backbone.startswith("swin"):
        transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ]
                                            )
        transform_test_flip = transforms.Compose([
            transforms.Resize((448, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        model = swin_t(num_classes=dataset.num_train_pids, loss="triplet",
                       camera=dataset.num_train_cams, sequence=dataset.num_train_seqs,
                       side_info=True,
                       version=params.backbone[-2:]).cuda()
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model.eval()
    if params.ckpt.endswith("pt") or params.ckpt.endswith("pth"):
        model.load_state_dict(torch.load(params.ckpt), strict=False)
    else:
        # providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(params.ckpt, providers=providers)
    # experimental
    market_gallery = reidDataset(dataset.gallery, dataset.num_train_pids, transform_test, False)
    dataloader1 = DataLoaderX(market_gallery, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    market_gallery.transform = transform_test_flip
    dataloader2 = DataLoaderX(market_gallery, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    gallery_embeddings, gallery_labels, gallery_cams, gallery_seqs = inference_efficient(model, dataloader1, dataloader2, dataset.num_gallery_cams, params.use_side)
    gallery_embeddings1 = gallery_embeddings[0]
    gallery_embeddings2 = gallery_embeddings[1]
    # gallery_embeddings1 = F.normalize(gallery_embeddings1, dim=1)
    # gallery_embeddings2 = F.normalize(gallery_embeddings2, dim=1)

    gallery_embeddings = (gallery_embeddings1 + gallery_embeddings2) / 2.0
    gallery_embeddings = F.normalize(gallery_embeddings, dim=1)

    market_query = reidDataset(dataset.query, dataset.num_train_pids, transform_test, False)
    dataloader1 = DataLoaderX(market_query, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    market_query.transform = transform_test_flip
    dataloader2 = DataLoaderX(market_query, batch_size=params.bs, num_workers=4, shuffle=False,
                              pin_memory=True)
    query_embeddings, query_labels, query_cams, query_seqs = inference_efficient(model, dataloader1, dataloader2,
                                                                                 dataset.num_query_cams,
                                                                                 params.use_side)
    query_embeddings1 = query_embeddings[0]
    query_embeddings2 = query_embeddings[1]
    # query_embeddings1 = F.normalize(query_embeddings1, dim=1)
    # query_embeddings2 = F.normalize(query_embeddings2, dim=1)

    query_embeddings = (query_embeddings1 + query_embeddings2) / 2.0
    query_embeddings = F.normalize(query_embeddings, dim=1)

    merged_embeddings = torch.cat((gallery_embeddings, query_embeddings), dim=0)
    merged_cams = torch.cat((gallery_cams, query_cams), dim=0)
    merged_seqs = torch.cat((gallery_seqs, query_seqs), dim=0)

    # from losses.utils import euclidean_dist
    dists = compute_jaccard_distance(merged_embeddings, search_option=0, use_float16=True)# euclidean_dist(merged_embeddings, merged_embeddings) #
    dists[dists < 0] = 0.
    dists[dists > 1] = 1.
    try:
        from cuml import DBSCAN
        print("CUML Imported!")
        cluster_method = DBSCAN(eps=params.eps, min_samples=min(12, dataset.num_gallery_cams+1), metric="precomputed")
        dists = dists.cpu().numpy()
    except ImportError:
        from sklearn.cluster import DBSCAN
        cluster_method = DBSCAN(eps=params.eps, min_samples=min(12, dataset.num_gallery_cams+1), metric="precomputed", n_jobs=-1)
    pseudo_labels = cluster_method.fit_predict(dists)
    indices_pseudo = (pseudo_labels != -1)
    num_labels = max(pseudo_labels) + 1
    assert num_labels >= 0.2 * dataset.num_train_pids
    # merged_seqs = merged_seqs * dataset.num_gallery_cams * num_labels + merged_cams * num_labels + pseudo_labels
    merged_seqs = merged_seqs * num_labels + pseudo_labels

    merged_embeddings = diminish_camera_bias(merged_embeddings, merged_cams)

    from inference_utils import smooth_tracklets
    merged_embeddings = smooth_tracklets(merged_embeddings, merged_seqs, indices_pseudo)

    gallery_embeddings = merged_embeddings[:len(dataset.gallery)]
    query_embeddings = merged_embeddings[len(dataset.gallery):]

    CMC = torch.IntTensor(gallery_embeddings.size(0)).zero_()
    ap = 0.0
    for i in range(query_embeddings.size(0)):
        ap_tmp, CMC_tmp = evaluate(query_embeddings[i],
                                   query_labels[i],
                                   query_cams[i],
                                   gallery_embeddings,
                                   gallery_labels,
                                   gallery_cams)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / query_embeddings.size(0)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / query_embeddings.size(0)))
