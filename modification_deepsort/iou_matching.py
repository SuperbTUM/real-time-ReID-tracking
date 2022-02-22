from __future__ import absolute_import
import numpy as np


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    bbox_center = np.asarray([(bbox_tl[1] + bbox_br[1])/2, (bbox_tl[0] + bbox_br[0])/2])
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    # In (x, y) style
    candidates_center = np.asarray([(candidates_tl[:,1] + candidates_br[:,1])/2,
                         (candidates_tl[:,0] + candidates_br[:,0])/2]).T
    d = np.sum((bbox_center - candidates_center) ** 2, axis=1)
    outer_tl = np.c_[np.minimum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                     np.minimum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    outer_br = np.c_[np.maximum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                     np.maximum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    rou = np.sum((outer_tl - outer_br) ** 2, axis=1)
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    iou = area_intersection / (area_bbox + area_candidates - area_intersection)
    return iou - d / rou


if __name__ == "__main__":
    bbox = np.asarray([10, 12, 8, 9])
    candidates = np.asarray([[9,10,9,9],[8,12,9,10],[10,12,9,8]])
    print(iou(bbox, candidates))