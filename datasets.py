import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, x, y):
        super(Dataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, item):
        xi = torch.tensor(self.x[item])
        yi = torch.tensor(self.y[item])
        return xi, yi

    def __len__(self):
        return len(self.x)


def datagenerator(xPath, yPath):
    dataX = []
    dataY = []

    xFiles = os.listdir(r"./" + xPath)
    yFiles = os.listdir(r"./" + yPath)
    for filename in xFiles:
        img = Image.fromarray(cv2.imread(xPath + "/" + filename))
        # img = cv2.imread(xPath + "/" + filename)
        try:
            img = np.asarray(F.resize(img, size=[256, 256])).transpose([2,0,1])
        except():
            print(Exception)
            continue
        dataX.append(img)
        print(filename)
    for filename in yFiles:
        img = Image.fromarray(cv2.imread(yPath + "/" + filename,flags=cv2.IMREAD_GRAYSCALE))
        # img = cv2.imread(yPath + "/" + filename)
        try:
            img = np.asarray(F.resize(img, size=[256, 256], interpolation=F.InterpolationMode.NEAREST))
        except():
            print(Exception)
            continue
        dataY.append(img)
        print(filename)

    dataX = np.array(dataX).astype('float32')/255.0
    dataY = np.array(dataY).astype('int64')
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.15)
    return trainDataX, trainDataY, testDataX, testDataY
