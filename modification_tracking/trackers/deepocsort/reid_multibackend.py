try:
    from yolov8.ultralytics.yolo.utils.checks import check_requirements, check_version
    from yolov8.ultralytics.yolo.utils import LOGGER
except:
    from yolov5.utils.general import check_requirements, check_version, LOGGER