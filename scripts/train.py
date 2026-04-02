import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from src.data import get_dataloader
from src.model import SegmentationModel
from scripts.val import validate, IoU, Dice


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    def forward(self, preds, labels):
        dice_loss = self.dice_loss(preds, labels)
        bce_loss = self.bce_criterion(preds, labels)
        loss = 0.4 * bce_loss + 0.6 * dice_loss
        return loss

    def dice_loss(self, preds, labels):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        labels = labels.view(-1)
        intersection = (preds * labels).sum()
        dice = (2. * intersection + 1) / (preds.sum() + labels.sum() + 1)
        return 1 - dice

def train_model(model, train_loader, val_loader, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = ComboLoss()

    losses = []

    best_dice = 0.0



    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_iou = total_dice = 0.0

        for img, label in train_loader:
            # img = img.to(device)
            # label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            total_iou += IoU(output.detach(), label)
            total_dice += Dice(output.detach(), label)
        
        print(f"Epoch {epoch + 1}/{epochs}")

        n_train = len(train_loader)
        train_loss = running_loss / n_train
        train_iou = total_iou / n_train
        train_dice = total_dice / n_train

        losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train IoU: {train_iou:.4f}")
        print(f"Train Dice: {train_dice:.4f}")

        val_iou, val_dice = validate(model, val_loader)

        print(f"Val IoU: {val_iou:.4f}")
        print(f"Val Dice: {val_dice:.4f}")

        if (val_dice > best_dice):
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')    

    torch.save(model.state_dict(), 'last_model.pth')
    build_graph_loss(losses)




def build_graph_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Train Loss', color='red', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_graph.png', dpi=150)


def main():
    set_seed(42)
    root_dir = "./data/massachusetts-roads-dataset/tiff/"

    train_loader = get_dataloader(root_dir, "/train", True)
    val_loader = get_dataloader(root_dir, "/val", False)

    model = SegmentationModel(64)
    train_model(model, train_loader, val_loader, 10)


if __name__ == "__main__":
    main()