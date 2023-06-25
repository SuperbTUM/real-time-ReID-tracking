import torch


def diminish_camera_bias(embeddings, cams, la=0.005):
    num_cams = cams.max().int()
    for i in range(num_cams+1):
        cur_embeddings = embeddings[cams == i]
        cam_bias = cur_embeddings.mean(dim=0)
        embeddings[cams == i] -= cam_bias
        tmp_eye = torch.eye(embeddings.shape[1])
        P = torch.inverse(cur_embeddings.T.matmul(cur_embeddings) + cur_embeddings.shape[0] * la * tmp_eye)
        embeddings[cams == i] = embeddings[cams == i].matmul(P.t())
        embeddings[cams == i] = embeddings[cams == i]/torch.norm(embeddings[cams == i], p=2, dim=1).unsqueeze(1)
    return embeddings
