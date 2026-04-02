import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloader
from src.model import SegmentationModel

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

import numpy as np


def IoU(preds, labels, smooth=1e-7):
    preds = torch.sigmoid(preds)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def Dice(preds, labels, smooth=1e-7):
    preds = torch.sigmoid(preds)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def validate(model, val_loader):
    model.eval()
    total_iou = total_dice = 0.0

    with torch.no_grad():
        for img, label in val_loader:
            # img = img.to(device)
            # label = label.to(device)
            output = model(img)
            total_iou += IoU(output, label)
            total_dice += Dice(output, label)

    n_val = len(val_loader)
    iou = total_iou / n_val
    dice = total_dice / n_val

    return iou, dice


def main():
    root_dir = "./data/massachusetts-roads-dataset/tiff/"

    val_loader = get_dataloader(root_dir, "/val")

    model = SegmentationModel(16)
    model.load_state_dict(torch.load("best_model.pth"))
    iou, dice = validate(model, val_loader)

    print(f"Val IoU:  {iou:.4f}")
    print(f"Dice: {dice:.4f}")


if __name__ == "__main__":
    main()