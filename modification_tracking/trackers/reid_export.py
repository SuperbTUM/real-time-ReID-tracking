try:
    from yolov8.ultralytics.yolo.utils import LOGGER, colorstr, ops
    from yolov8.ultralytics.yolo.utils.checks import check_requirements, check_version
except:
    from yolov5.utils.general import LOGGER, colorstr, ops, check_requirements, check_version