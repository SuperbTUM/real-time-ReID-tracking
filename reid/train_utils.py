from train_prepare import *
import torch.onnx
import torch.distributed as dist
import h5py
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

from accelerate import Accelerator
from ultralytics import YOLO


def transform_dataset_hdf5(gt_paths, img_width, img_height):
    """With OpenCV"""
    h5file = "import_images.h5"
    with h5py.File(h5file, "w") as h5f:
        image_ds = h5f.create_dataset("images", shape=(len(gt_paths), img_width, img_height, 3), dtype=int)
        for cnt, path in enumerate(gt_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img, (img_width, img_height))
            image_ds[cnt:cnt+1, :, :] = img_resize
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


def accelerate_train(*args):
    accelerator = Accelerator()
    return {"accelerated": accelerator.prepare(*args), "accelerator":  accelerator}


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


def redetection(image, format="pil", conf=0.3):
    model = YOLO("yolov8n.pt")
    result = model(image, verbose=False, device=0)
    bbox = None
    for r in result:
        boxes = r.boxes
        if boxes:
            for klass, konf in zip(boxes.cls, boxes.conf):
                if klass.item() == 0 and konf.item() > conf:
                    conf = konf.item()
                    bbox = boxes.xyxy[0]
    if bbox is not None:
        if format == "pil":
            width, height = image.size
            image = np.array(image)
        else:
            height, width = image.shape[:2]
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(width, bbox[2]))
        y2 = int(min(height, bbox[3]))
        image = image[y1:y2, x1:x2, :]
        if format == "opencv":
            return image # np.ndarray
        elif format == "pil":
            return Image.fromarray(image)
        else:
            raise NotImplementedError
    return image
