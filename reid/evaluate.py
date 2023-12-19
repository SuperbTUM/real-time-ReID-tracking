import numpy as np
import torch


def get_feats(model, dataloader):
    model.eval()
    labels = []
    features = []
    cams = []
    for sample in dataloader:
        image, label, cam = sample[:3]
        feat = model(image.cuda(), cam)
        features.append(feat[1])
        labels.append(label)
        cams.append(cam)
    labels = torch.cat(labels, dim=0)
    features = torch.cat(features, dim=0)
    cams = torch.cat(cams, dim=0)
    model.train()
    return labels, features, cams


def get_evaluate_result(model, dataloader_query, dataloader_gallery):
    labels_query, features_query, cams_query = get_feats(model, dataloader_query)
    labels_gallery, features_gallery, cams_gallery = get_feats(model, dataloader_gallery)
    CMC, ap = evaluate_all(features_query, labels_query, cams_query, features_gallery, labels_gallery, cams_gallery)
    return CMC, ap


# @credit to Zhedong
#######################################################################
# Evaluate
def evaluate_all(qf, ql, qc, gf, gl, gc):
    CMC = torch.IntTensor(gf.size(0)).zero_()
    ap = 0.0
    for i in range(qf.size(0)):
        ap_tmp, CMC_tmp = evaluate(qf[i],
                                   ql[i],
                                   qc[i],
                                   gf,
                                   gl,
                                   gc)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC /= qf.size(0)  # average CMC
    ap /= qf.size(0)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap))
    return CMC, ap


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
