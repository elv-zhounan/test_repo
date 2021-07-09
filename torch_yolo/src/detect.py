import config
import torch
import torch.optim as optim
from PIL import Image
import os
import numpy as np
from model import YOLOv3
from tqdm import tqdm
from utils import (
    cells_to_bboxes,
    load_checkpoint,
    non_max_suppression
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# torch.cuda.set_device(1)

def plot_image(image_origin, boxes, pad_h, pad_w):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("Set1")
    class_labels = config.CUSTOM_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    
    origin_h, origin_w, _ = image_origin.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image_origin)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (int((upper_left_x * config.IMAGE_SIZE - pad_w) * (origin_w / (config.IMAGE_SIZE - 2*pad_w))), 
             int((upper_left_y * config.IMAGE_SIZE - pad_h) * (origin_h / (config.IMAGE_SIZE - 2*pad_h)))),
            box[2] * (origin_w * config.IMAGE_SIZE / (config.IMAGE_SIZE - 2*pad_w)),
            box[3] * (origin_h * config.IMAGE_SIZE / (config.IMAGE_SIZE - 2*pad_h)),
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()


def load_model():
    # load model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    print("=> Loading checkpoint")
    checkpoint = torch.load(config.EVAL_CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Loaded")

    return model

def pred(model, img_path, scaled_anchors):
    # img_path = '/Users/zhounanli/Elv_logo_project/LogosInTheWild-v2/data_cleaned/data/custom/images/train/adidas_img000182_4.jpg'
    label_path = "labels".join(img_path.rsplit("images", 1))
    label_path = os.path.splitext(label_path)[0] + '.txt'

    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1)
    bboxes[:, 2:4] = bboxes[:, 2:4] - 0.001
    bboxes = bboxes.tolist()
    image = np.array(Image.open(img_path).convert("RGB"))

    transform = config.test_transforms

    augmentations = transform(image=image, bboxes=bboxes)
    image_t = augmentations["image"]

    image_t = image_t.view(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    with torch.no_grad():
        out = model(image_t)
        bboxes = [[] for _ in range(image_t.shape[0])]
        for i in range(3):
            batch_size, _, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

    # pred res
    nms_boxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.6, box_format="midpoint",)

    # get some tranform infomation
    h, w, c = image.shape
    pad_h = 0
    pad_w = 0

    if w > h:
        pad_h = int((w - h) / (w / config.IMAGE_SIZE) / 2)
    else:
        pad_w = int((h - w) / (h / config.IMAGE_SIZE) / 2)


    return nms_boxes, pad_h, pad_w, image




if __name__ == "__main__":
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    model = load_model()

    img_path = '/Users/zhounanli/Elv_logo_project/LogosInTheWild-v2/data_cleaned/data/custom/images/val/aldi_img000015_6.jpg'

    nms_boxes, pad_h, pad_w, image_origin = pred(model, img_path, scaled_anchors)

    plot_image(image_origin, nms_boxes, pad_h, pad_w)