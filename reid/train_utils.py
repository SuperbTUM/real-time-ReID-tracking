from train_prepare import *
import torch.onnx
import torch.distributed as dist
import h5py
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import onnxruntime

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


def export_yolo(sz=(256, 128)):
    model = YOLO("yolov8n.pt")
    if os.path.exists("yolov8n.onnx"):
        return True
    success = model.export(format="onnx", imgsz=sz, dynamic=True, device=0)
    return success


def redetection(image, format="pil", base_conf=0.5):
    """
    batched detection
    """
    # # success = model.export(format="onnx")
    # processed_image = np.array(image, dtype=np.float32) / 255.
    # processed_image = np.expand_dims(processed_image.transpose((2, 0, 1)), axis=0)
    # providers = ["CUDAExecutionProvider"]
    # ort_session = onnxruntime.InferenceSession("yolov8n.onnx", providers=providers)
    # model_inputs = ort_session.get_inputs()
    # model_outputs = ort_session.get_outputs()
    # input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    # output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    # outputs = np.squeeze(ort_session.run(output_names, {input_names[0]: processed_image})[0]).T
    model = YOLO("best.pt")  # credit to @jahongir7174
    results = model(image, imgsz=(256, 128), classes=0, verbose=False)
    bbox = None
    for result in results[0]:
        outputs = result.boxes.data
        for output in outputs:
            klass = output[-1].item()
            konf = output[-2].item()
            if klass == 0 and konf > base_conf:
                base_conf = konf
                bbox = output[:4]

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
    return image


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
