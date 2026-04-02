import os
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import transforms


class RoadDataset(Dataset):
    def __init__(self, imgs, labels, img_transform=None, lbl_transform=None):

        self.imgs = imgs
        self.labels = labels
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = self.imgs[idx]
        lbl_path = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('L')
        # img = img.resize((512, 512))
        # lbl = lbl.resize((512, 512))
        input_img = self.img_transform(img)
        # input_img = F.normalize(input_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        input_lbl = self.lbl_transform(lbl)
        # input_lbl = input_lbl[0:1, :, :]
        input_lbl = (input_lbl >= 0.5).float()

        return input_img, input_lbl


def get_dataloader(root_dir, phase, shuffle):
    images, labels = [], []
    photos = sorted([d for d in os.listdir(root_dir + phase)])
    photos_lab = sorted([d for d in os.listdir(root_dir + phase + "_labels")])

    for photo in photos:
        img_path = os.path.join(root_dir + phase, photo)
        images.append(img_path)

    for photo in photos_lab:
        lbl_path = os.path.join(root_dir + phase + "_labels", photo)
        labels.append(lbl_path)

    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    lbl_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    test_dataset = RoadDataset(images, labels, img_transform, lbl_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=shuffle)

    return test_loader
