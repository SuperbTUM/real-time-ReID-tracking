import json
import pandas as pd
from collections import defaultdict


def MOT16_txt2dict(path):
    MOT16_gt = pd.read_csv(path, header=None)
    MOT16_gt = MOT16_gt[(MOT16_gt[6] == 1) & (MOT16_gt[7] == 1)]
    MOT16_gt_set = set(MOT16_gt[0])
    MOT16_dict = {}

    for unique_key in MOT16_gt_set:
        temp_list = []
        for index, row in MOT16_gt[MOT16_gt[0] == unique_key].iterrows():
            values = list(map(lambda x: int(x), row[[2, 3, 4, 5]]))
            person_id = int(row[1])
            temp_list.append([person_id] + values)

        MOT16_dict[unique_key] = temp_list

    with open("MOT16_05_dict.json", "w") as f:
        json.dump(MOT16_dict, f)
    f.close()


def load_gt(path):
    with open(path, "r") as f:
        ground_truth = json.load(f)
    return ground_truth


def load_tracking_res(path):
    MOT16_tracking = pd.read_csv(path, header=None).values
    MOT16_dict = defaultdict(list)
    for tracking_details in MOT16_tracking:
        tracking_details = tracking_details[0].strip().split()
        frame = int(tracking_details[0])
        id = int(tracking_details[1])
        bbox = list(map(lambda x: int(x), tracking_details[2:6]))
        MOT16_dict[frame].append([id] + bbox)
    with open("MOT16_05_nn.json", "w") as f:
        json.dump(MOT16_dict, f)
    f.close()


# unfinished
def MOTA(baseline_tracking, gan_tracking, ground_truth):
    with open(ground_truth, "r") as gt:
        ground_truth = json.load(gt)
    gt.close()
    with open(baseline_tracking, "r") as bt:
        baseline_tracking = json.load(bt)
    bt.close()
    with open(gan_tracking, "r") as gan:
        gan_tracking = json.load(gan)
    gan.close()
    false_detection_baseline = 0
    id_switch_baseline = 0
    false_detection_gan = 0
    id_switch_gan = 0
    GT = 0
    for i in range(3, 838):
        gt_frame = ground_truth[i]
        baseline_frame = baseline_tracking[i]
        gan_frame = gan_tracking[i]
        if len(gan_frame) > len(baseline_frame):
            print(i)
    return
