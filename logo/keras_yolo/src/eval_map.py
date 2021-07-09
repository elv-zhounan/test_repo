import os
import argparse
import random

parser = argparse.ArgumentParser(description='main')
parser.add_argument("--detect_thres", type=float, default = 0.1)
parser.add_argument("--gpu_id", type=str, default = "0")
parser.add_argument("--subset", type=int, default = 50)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import json
from PIL import Image
import imageio
import tqdm
import utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.python.keras.backend as K
from yolo_detection import YOLO

# def _build_session(graph):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
#     sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#     tf.Session(config=sess_config, graph=graph)
#     return

# _build_session(tf.Graph())





if __name__ == '__main__':


    # load model
    yolo = YOLO(score=args.detect_thres)


    with open("../data/custom/valid.txt") as file:
        file_names = file.readlines()

    pred_labels = [] # [sample_idx, cls, score, x1, y1, x2, y2]
    true_labels = [] # [sample_idx, cls, score, x1, y1, x2, y2]
    if args.subset > 0:
        file_names_subset = random.choices(file_names, k=args.subset)
    else:
        file_names_subset = file_names

    for idx in tqdm.tqdm(range(len(file_names_subset))):
        img_path = file_names_subset[idx][:-1]
        label_path = "labels".join(img_path.rsplit("images", 1)) 
        label_path = os.path.splitext(label_path)[0] + '.txt'
        # load img
        img = Image.open(img_path)
        w, h = img.size

        # load ground_truth_bbox
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1)
        bboxes[:, 2:4] = bboxes[:, 2:4] - 0.0001  # [x, y, w, h]
        bboxes = bboxes.tolist()
        # add to true_labels list
        for box in bboxes:
            true_labels.append([idx, 0, 1] + box)

        # detect
        preds = yolo.detect_image(img)  # [x0, y0, x1, y1, cls, score]
        for (x0, y0, x1, y1, c, s) in preds:

            pred_labels.append([idx, 0, s, (x0/w+x1/w)/2, (y0/h+y1/h)/2, x1/w-x0/w, y1/h-y0/h])
    

    AP = utils.mean_average_precision(pred_labels, true_labels)
    print(AP)

