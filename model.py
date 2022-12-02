# Import necessary packages.
import numpy as np
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
import random


class ResBlockA(nn.Module):

    def __init__(self, in_chann, chann, stride):
        super(ResBlockA, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        
        if (x.shape == y.shape):
            z = x
        else:
            z = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)            

            x_channel = x.size(1)
            y_channel = y.size(1)
            ch_res = (y_channel - x_channel)//2

            pad = (0, 0, 0, 0, ch_res, ch_res)
            z = nn.functional.pad(z, pad=pad, mode="constant", value=0)

        z = z + y
        z = nn.functional.relu(z)
        return z


class BaseNet(nn.Module):
    
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # [64, 16, 128, 128]
        self.bn0   = nn.BatchNorm2d(16) # [64, 16, 128, 128]
        self.convs  = self._make_layers(n) # [64, 64, 32, 32]
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1) # [64, 64, 25, 25]
        self.fc = nn.Linear(64*9*9, 13)

    def forward(self, x):
        x = self.conv0(x)
        # print(x.shape)
        x = self.bn0(x)
        # print(x.shape)
        x = nn.functional.relu(x)
        # print(x.shape)
        x = self.convs(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 16
        chann = 16
        stride = 1
        for i in range(3):
            for j in range(n):
                if ((i > 0) and (j == 0)):
                    in_chann = chann
                    chann = chann * 2
                    stride = 2

                layers += [self.Block(in_chann, chann, stride)]

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)


def Classifier(n):
    return BaseNet(ResBlockA, n)


class MyModel:
     # 建構式
    def __init__(self, n_layer, path):
        # "cuda" only when GPUs are available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize a model, and put it on the device specified.
        self.model = Classifier(6).to(self.device)
        self.model.load_state_dict(torch.load(path))

        self.tfm = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    
    def pred(self, img):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.tfm(img)
        img = img.reshape(1, 3, 64, 64).to(self.device)
        logits = self.model(img)
        _,pred = torch.max(logits, 1)
        return str(pred.item())

