import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloader
from src.model import SegmentationModel
from scripts.val import validate

import torch


def main():
    test_dir = "./data/massachusetts-roads-dataset/tiff/"

    test_loader = get_dataloader(test_dir, "/test", False)

    model = SegmentationModel(64)
    model.load_state_dict(torch.load("best_model.pth"))
    iou, dice = validate(model, test_loader)

    print(f"Test IoU:  {iou:.4f}")
    print(f"Test Dice: {dice:.4f}")


if __name__ == "__main__":
    main()
