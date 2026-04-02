import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationModel(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Энкодер
        self.conv1 = DoubleConv(3, nc)
        self.conv2 = DoubleConv(nc, 2 * nc)
        self.conv3 = DoubleConv(2 * nc, 4 * nc)
        self.conv4 = DoubleConv(4 * nc, 8 * nc)
        self.conv5 = DoubleConv(8 * nc, 16 * nc)

        self.unconv4 = nn.ConvTranspose2d(16 * nc, 8 * nc, 2, 2)
        self.conv6 = DoubleConv(16 * nc, 8 * nc)  # 8+8 = 16

        self.unconv3 = nn.ConvTranspose2d(8 * nc, 4 * nc, 2, 2)
        self.conv7 = DoubleConv(8 * nc, 4 * nc)  # 4+4 = 8

        self.unconv2 = nn.ConvTranspose2d(4 * nc, 2 * nc, 2, 2)
        self.conv8 = DoubleConv(4 * nc, 2 * nc)  # 2+2 = 4

        self.unconv1 = nn.ConvTranspose2d(2 * nc, nc, 2, 2)
        self.conv9 = DoubleConv(2 * nc, nc)      # 1+1 = 2

        self.conv10 = nn.Conv2d(nc, 1, 1)

    def forward(self, x):

        x0 = self.conv1(x) 
        x1 = self.conv2(self.max_pool(x0))
        x2 = self.conv3(self.max_pool(x1))
        x3 = self.conv4(self.max_pool(x2))
        x4 = self.conv5(self.max_pool(x3))

        # Вверх
        up_4 = self.unconv4(x4)
        cat_4 = torch.cat((x3, up_4), dim=1)
        rev_4 = self.conv6(cat_4)

        up_3 = self.unconv3(rev_4)
        cat_3 = torch.cat((x2, up_3), dim=1)
        rev_3 = self.conv7(cat_3)

        up_2 = self.unconv2(rev_3)
        cat_2 = torch.cat((x1, up_2), dim=1)
        rev_2 = self.conv8(cat_2)

        up_1 = self.unconv1(rev_2)
        cat_1 = torch.cat((x0, up_1), dim=1)
        rev_1 = self.conv9(cat_1)

        return self.conv10(rev_1)
