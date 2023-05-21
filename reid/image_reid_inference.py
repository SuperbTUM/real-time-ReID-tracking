from dataset_market import Market1501
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

from train_utils import to_numpy, DataLoaderX
from image_reid_train import MarketDataset

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
                    embeddings, _ = model(img, cam + all_cam * seq)
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


def parser():
    args = argparse.ArgumentParser()
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
                                                                            "resnet50",
                                                                            "baseline"])
    args.add_argument("--use_side", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    dataset = Market1501(root="/".join((params.root, "Market1501")))

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
        if params.backbone == "seres18":
            model = seres18_ibn(num_classes=dataset.num_train_pids, loss="triplet", renorm=True).cuda()
        else:
            model = cares18_ibn(dataset.num_train_pids, renorm=True).cuda()
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
        providers = ["CUDAExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(params.ckpt, providers=providers)
    market_gallery = MarketDataset(dataset.gallery, transform_test, False)
    dataloader = DataLoaderX(market_gallery, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    gallery_embeddings, gallery_labels, gallery_cams = inference(model, dataloader, dataset.num_gallery_cams, True
    if params.ckpt.endswith("onnx") else False, params.use_side)
    gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
    market_gallery_augment = MarketDataset(dataset.gallery, transform_test_flip, False)
    dataloader = DataLoaderX(market_gallery_augment, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    gallery_embeddings_augment, _, _ = inference(model, dataloader, dataset.num_gallery_cams, True
    if params.ckpt.endswith("onnx") else False, params.use_side)
    gallery_embeddings_augment = F.normalize(gallery_embeddings_augment, dim=1)

    gallery_embeddings = (gallery_embeddings + gallery_embeddings_augment) / 2.0

    market_query = MarketDataset(dataset.query, transform_test, False)
    dataloader = DataLoaderX(market_query, batch_size=params.bs, num_workers=4, shuffle=False, pin_memory=True)
    query_embeddings, query_labels, query_cams = inference(model, dataloader, dataset.num_query_cams, True
    if params.ckpt.endswith("onnx") else False, params.use_side)
    query_embeddings = F.normalize(query_embeddings, dim=1)
    market_query_augment = MarketDataset(dataset.query, transform_test_flip, False)
    dataloader = DataLoaderX(market_query_augment, batch_size=params.bs, num_workers=4, shuffle=False,
                             pin_memory=True)
    query_embeddings_augment, _, _ = inference(model, dataloader, dataset.num_query_cams, True
    if params.ckpt.endswith("onnx") else False, params.use_side)
    query_embeddings_augment = F.normalize(query_embeddings_augment, dim=1)

    query_embeddings = (query_embeddings + query_embeddings_augment) / 2.0

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
