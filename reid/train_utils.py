import numpy as np

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch.onnx
import torch.distributed as dist
import h5py
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from segmentation import batched_extraction

from ultralytics import YOLO


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def transform_dataset_hdf5(gt_paths, img_width, img_height):
    """With OpenCV"""
    h5file = "import_images.h5"
    with h5py.File(h5file, "w") as h5f:
        image_ds = h5f.create_dataset("images", shape=(len(gt_paths), img_width, img_height, 3), dtype=int)
        for cnt, path in enumerate(gt_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img, (img_width, img_height))
            image_ds[cnt:cnt + 1, :, :] = img_resize
    return image_ds


def recover_from_hdf5(image_ds, index):
    with h5py.File(image_ds, "r") as h5f:
        # Random access
        image = h5f["images"][index, ...]
    return image


def ddp_trigger(model, rank=-1, world_size=-1):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    assert dist.is_available()
    from torch.nn.parallel import DistributedDataParallel as DDP

    def setup(rank=rank, world_size=world_size):
        if dist.is_nccl_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    setup(rank, world_size)

    """Original training
    """
    # Distributed training
    # ----------
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.train()
    return ddp_model


def postprocess_ddp(ddp_model, rank):
    def cleanup():
        dist.destroy_process_group()

    if rank == 0:
        torch.save(ddp_model.state_dict(), "checkpoint.pt")
    dist.barrier()
    cleanup()


def plot_loss(loss_stats):
    plt.figure()
    plt.plot(np.arange(len(loss_stats)), loss_stats, linewidth=2, label="train loss")
    plt.xlabel("iterations")
    plt.ylabel('loss')
    plt.title('training loss')
    plt.legend()
    plt.grid()
    if not os.path.exists("images/"):
        os.mkdir("images/")
    plt.savefig("images/loss_curve.png")
    plt.show()


def export_yolo(sz=(256, 128)):
    model = YOLO("yolov8n.pt")
    if os.path.exists("yolov8n.onnx"):
        return True
    success = model.export(format="onnx", imgsz=sz, dynamic=True, device=0)
    return success


model = torch.hub.load('ultralytics/yolov5', "custom", path="crowdhuman_yolov5m.pt", source="local", _verbose=False)


def redetection(images, format="pil", base_conf=0.5):
    """
    batched detection
    """
    result = model(images, size=(256, 128), augment=True)
    result = result.xyxy
    bboxes = []
    for res in result:
        bbox = None
        conf = base_conf
        for r in res:
            klass = r[-1]
            konf = r[-2]
            if klass.item() == 0 and konf.item() > conf:
                conf = konf.item()
                bbox = r[:4]
        bboxes.append(bbox)
    # for output in outputs:
    #     klasses = np.argmax(output[:, 4:], axis=1)
    #     konfs = np.max(output[:, 4:], axis=1)
    #     for klass, konf in zip(klasses, konfs):
    #         if klass == 0 and konf > conf:
    #             conf = konf
    #             bbox = output[klass, :4]
    images_cropped = []
    for bbox, image in zip(bboxes, images):
        if bbox is not None:
            if format == "pil":
                width, height = image.size
            else:
                height, width = image.shape[:2]
            x1 = int(max(0, bbox[0]))
            y1 = int(max(0, bbox[1]))
            x2 = int(min(width, bbox[2]))
            y2 = int(min(height, bbox[3]))
            if format == "opencv":
                image = image[y1:y2, x1:x2, :]
            elif format == "pil":
                image = image.crop((x1, y1, x2, y2))
            else:
                raise NotImplementedError
        images_cropped.append(image)
    return images_cropped


def recrop(images, format="pil", base_conf=0.5):
    foregrounds = batched_extraction(images, blured=True)[0]
    if format == "pil":
        fs = []
        for foreground in foregrounds:
            foreground = Image.fromarray(foreground)
            fs.append(foreground)
        return fs
    return foregrounds


def check_parameters(model):
    # credit to https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def mixup_data(x, y, alpha=.99, intra_only=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        distri = torch.distributions.beta.Beta(alpha, alpha)
        lam = distri.sample().item()
    else:
        lam = alpha

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    if intra_only:
        mixed_x = []
        for i, (y_a, y_b) in enumerate(zip(y, y[index])):
            if y_a == y_b:
                mixed_x.append(x[i])
            else:
                mixed_x.append(lam * x[i] + (1 - lam) * x[index[i]])
        mixed_x = torch.stack(mixed_x)
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
