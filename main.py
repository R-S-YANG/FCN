import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datasets
import model
import utils.indicators as indicators
######################### Hyperparameter ############################
BATCH_SIZE = 20  # 每批处理数据的数量
EPOCHS = 100  # 数据集训练的轮次
LEARNING_RATE = 10e-3  # 学习率
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 使用gpu还是cpu
SAVE_DIR = os.path.join('weights')
########################## Datasets #################################
trainDataX, trainDataY, testDataX, testDataY = datasets.datagenerator('weizmann_horse_db/horse',
                                                                      'weizmann_horse_db/mask')
trainDataset = datasets.Datasets(trainDataX, trainDataY)
testDataset = datasets.Datasets(testDataX, testDataY)
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE, shuffle=True)
print("trainX:", len(trainDataX), "\n", trainDataX.shape)
print("trainY:", len(trainDataY), "\n", trainDataY.shape)
print("testX:", len(testDataX), "\n", testDataX.shape)
print("testY:", len(testDataY), "\n", testDataY.shape)

if __name__ == '__main__':
    ########################## Training #################################
    model = model.FCN()
    model = model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        # 训练过程
        with tqdm(total=(len(trainDataset) - len(trainDataset) % BATCH_SIZE)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, EPOCHS))
            epoch_losses = 0
            cnt = 0
            for data in trainDataLoader:
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model(inputs)
                # print(preds.shape)
                # print(labels.shape)

                loss = criterion(preds, labels)
                epoch_losses += loss.item()
                cnt += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses / cnt))
                _tqdm.update(len(inputs))
            writer.add_scalar('train loss', loss, epoch)
        torch.save(model, os.path.join(SAVE_DIR, 'epoch{}.pth'.format(epoch + 1)))

        model.eval()
        test_loss=0
        correct = 0
        MIOU = []
        BIOU = []
        for index, (inputs, labels) in enumerate(testDataLoader):
            inputs, labels = inputs.to(DEVICE),labels.to(DEVICE)
            output = model(inputs).argmax(1)
            # preds = output.max(2, keepdim=True)
            # 求MIOU
            for i in range(len(output)):
                pred = output[i]
                target = labels[i]
                MIOU.append(indicators.MIOU(2, pred, target))
                BIOU.append(indicators.boundary_iou(target,pred))
        print('MIOU=',np.mean(MIOU))
        print('MIOU=',np.mean(BIOU))
    writer.close()
