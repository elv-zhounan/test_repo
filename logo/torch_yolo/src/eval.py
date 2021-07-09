import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

# torch.cuda.set_device(1)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model = torch.nn.DataParallel(model)

    train_loader, test_loader, train_eval_loader = get_loaders(
        "../data/custom/train.txt", "../data/custom/valid.txt"
    )


    print("=> Loading checkpoint")
    checkpoint = torch.load(config.EVAL_CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Loaded")
    

    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
        
    # print("=> checking accuracy for obj and non-obj")
    # check_(model, test_loader, threshold=config.CONF_THRESHOLD)
    # 0.9686, 0.7936
    
    print("=> checking AP for the only class")
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")


if __name__ == "__main__":
    main()
