import torch.nn
from torch import nn


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.layers1 = []
        self.layers2 = []
        self.layers3 = []

        self.layers1 = nn.Sequential(
            ## 第一个block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),  # 256*256*64
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),  # 256*256*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128*128*64
            ## 第二个block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),  # 128*128*128
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),  # 128*128*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*128
            ## 第三个block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),  # 64*64*256
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),  # 64*64*256
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),  # 64*64*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32*32*256
        )  # layers1输出y1为30*30*256

        self.layers2 = nn.Sequential(
            ## 第四个block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),  # 32*32*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 32*32*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 32*32*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16*16*512
        )
        ## 第五个block
        self.layers3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 16*16*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 16*16*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 16*16*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8*512
            # vgg的全连接改为卷积
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(3, 3), padding=1),  # 8*8*4096
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(3, 3), padding=1),  # 8*8*4096
            nn.ReLU(inplace=True),
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.upsample8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear")
        )
        self.afterConv32s = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=2, kernel_size=(3, 3), padding=1)
        )
        self.afterConv16s = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(3, 3), padding=1)
        )
        self.afterConv8s = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        # 获取原图的1/8，1/16和1/32三张特征图
        pool3 = self.layers1(x)
        pool4 = self.layers2(pool3)
        conv7 = self.layers3(pool4)
        # 融合
        fcn32s = self.upsample2(self.afterConv32s(conv7))  # 16*16*2
        fcn16s = self.upsample2(fcn32s + self.afterConv16s(pool4))  # 32*32*2
        out = self.upsample8(fcn16s + self.afterConv8s(pool3))  # 256*256*2
        return out
