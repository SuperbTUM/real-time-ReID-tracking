from train_prepare import *
import torch.onnx
from tqdm import tqdm
from torch.autograd import Variable

from plr_osnet import plr_osnet
from train_utils import *


def train_plr_osnet(dataset, batch_size=8, epochs=25, num_classes=517):
    model = plr_osnet(num_classes=num_classes, loss='triplet').cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_func = HybridLoss3(num_classes=num_classes)
    loss_stats = []
    for epoch in range(epochs):
        dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
        iterator = tqdm(dataloader)
        for sample in iterator:
            images, label = sample
            optimizer.zero_grad()
            images = images.cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            global_branch, local_branch, feat = model(images)
            loss1 = loss_func(feat[:2048], global_branch, label)
            loss2 = loss_func(feat[2048:], local_branch, label)
            loss = loss1 + loss2
            loss_stats.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            description = "epoch: {}, lr: {}, loss: {:.4f}".format(epoch, lr_scheduler.get_last_lr()[0], loss)
            iterator.set_description(description)
    model.eval()
    torch.save(model.state_dict(), "in_video_checkpoint.pt")
    to_onnx(model, torch.randn(batch_size, 3, 256, 128, requires_grad=True))
    return model, loss_stats
