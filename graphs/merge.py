import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size, padding=1,
                              bias=True)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, kernel_size=3))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class DRB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRB, self).__init__()
        self.conv_1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=2, bias=True, dilation=2)
        self.conv_2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=2, bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.conv_2(out)
        out = out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(avg_out)
        x2 = self.conv2(max_out)
        x = x1 + x2
        return self.sigmoid(x)


class Encoder2Conv(nn.Module):
    def __init__(self, in_ch):
        super(Encoder2Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, 3, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, in_ch * 4, 3, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out

class Decoder2Conv(nn.Module):
    def __init__(self, nFeat):
        super(Decoder2Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(nFeat * 4, nFeat * 2, 3, stride=2, padding=0, output_padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nFeat * 2, nFeat, 3, stride=2, padding=0, output_padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, 3, 3, padding=1, padding_mode='zeros')
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# Attention Guided HDR, AHDR-Net
class MERGE(nn.Module):
    def __init__(self, nChannel, nDenselayer, nFeat, growthRate):
        super(MERGE, self).__init__()
        self.nChannel = nChannel
        self.nDenselayer = nDenselayer
        self.nFeat = nFeat
        self.growthRate = growthRate

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # self.conv1_1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(nFeat)
        # self.conv1_2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(nFeat)
        # F0
        self.conv2_1 = nn.Conv2d(nFeat * 8, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(nFeat * 8, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat * 8, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.ca1 = ChannelAttention(nFeat * 2)
        self.sa1 = SpatialAttention()
        self.ca3 = ChannelAttention(nFeat * 2)
        self.sa3 = SpatialAttention()

        # DRDBs 3
        self.DRDB1_1 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB1_2 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB1_3 = DRB(nFeat, nDenselayer, growthRate)
        # self.DRDB1_4 = RB(nFeat, nDenselayer, growthRate)
        # self.DRDB1_5 = RB(nFeat, nDenselayer, growthRate)

        self.DRDB2_1 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB2_2 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB2_3 = DRB(nFeat, nDenselayer, growthRate)
        # self.DRDB2_4 = RB(nFeat, nDenselayer, growthRate)
        # self.DRDB2_5 = RB(nFeat, nDenselayer, growthRate)

        self.DRDB1 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB2 = DRB(nFeat, nDenselayer, growthRate)
        self.DRDB3 = DRB(nFeat, nDenselayer, growthRate)
        # self.DRDB4 = RB(nFeat, nDenselayer, growthRate)
        # self.DRDB5 = RB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1_1 = nn.Conv2d(nFeat * 3, nFeat * 4, kernel_size=1, padding=0, bias=True)
        self.GFF_1x1_2 = nn.Conv2d(nFeat * 3, nFeat * 4, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_1 = nn.Conv2d(nFeat, nFeat * 4, kernel_size=3, padding=1, bias=True)
        self.GFF_3x3_2 = nn.Conv2d(nFeat, nFeat * 4, kernel_size=3, padding=1, bias=True)

        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat * 4, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        #self.relu = nn.LeakyReLU()

        self.encoder1 = Encoder2Conv(nFeat)
        self.encoder2 = Encoder2Conv(nFeat)
        self.encoder3 = Encoder2Conv(nFeat)

        self.decoder = Decoder2Conv(nFeat)
    def forward(self, x1, x2, x3):
        # F1_ = self.conv1_1(x1)
        # F1_ = F.relu(self.bn1(F1_))
        # F1_ = self.conv1_2(F1_)
        # F1_ = self.bn2(F1_)
        # F2_ = self.conv1_1(x2)
        # F2_ = F.relu(self.bn1(F2_))
        # F2_ = self.conv1_2(F2_)
        # F2_ = self.bn2(F2_)
        # F3_ = self.conv1_1(x3)
        # F3_ = F.relu(self.bn1(F3_))
        # F3_ = self.conv1_2(F3_)
        # F3_ = self.bn2(F3_)
        F1_ = F.relu(self.conv1(x1))
        F2_ = F.relu(self.conv1(x2))
        F3_ = F.relu(self.conv1(x3))

        # F1_i = torch.cat((F1_, F2_), 1)
        # F1_A = F.relu(self.att11(F1_i))
        # F1_A = self.att12(F1_A)
        # F1_A = torch.sigmoid(F1_A)
        # F1_ = F1_ * F1_A

        # F3_i = torch.cat((F3_, F2_), 1)
        # F3_A = F.relu(self.att31(F3_i))
        # F3_A = self.att32(F3_A)
        # F3_A = torch.sigmoid(F3_A)
        # F3_ = F3_ * F3_A

        F1_i = torch.cat((F1_, F2_), 1)
        F1_C = self.ca1(F1_i)
        F1_C = F1_ * F1_C
        F1_A = self.sa1(F1_C)
        F1_ = F1_C * F1_A

        F3_i = torch.cat((F3_, F2_), 1)
        F3_C = self.ca3(F3_i)
        F3_C = F3_ * F3_C
        F3_A = self.sa3(F3_C)
        F3_ = F3_C * F3_A

        F1_ = self.encoder1(F1_)
        F2_ = self.encoder2(F2_)
        F3_ = self.encoder3(F3_)

        F_1 = torch.cat((F1_, F2_), 1)
        F_2 = torch.cat((F3_, F2_), 1)

        F_1_0 = F.relu(self.conv2_1(F_1))
        F_1_1 = self.DRDB1_1(F_1_0)
        F_1_2 = self.DRDB1_2(F_1_1)
        F_1_3 = self.DRDB1_3(F_1_2)
        FF_1 = torch.cat((F_1_1, F_1_2, F_1_3), 1)
        FdLF_1 = self.GFF_1x1_1(FF_1)
        FDF_1 = FdLF_1 + F2_

        F_2_0 = F.relu(self.conv2_2(F_2))
        F_2_1 = self.DRDB2_1(F_2_0)
        F_2_2 = self.DRDB2_2(F_2_1)
        F_2_3 = self.DRDB2_3(F_2_2)
        FF_2 = torch.cat((F_2_1, F_2_2, F_2_3), 1)
        FdLF_2 = self.GFF_1x1_2(FF_2)
        FDF_2 = FdLF_2 + F2_

        F_ = torch.cat((FDF_1, FDF_2), 1)

        # F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = F.relu(self.conv2(F_))
        F_1 = self.DRDB1(F_0)
        F_2 = self.DRDB2(F_1)
        F_3 = self.DRDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_ + F1_ + F3_
        # us = self.conv_up(FDF)
        #
        # output = self.conv3(us)
        output = self.decoder(FDF)
        output = torch.sigmoid(output)

        return output

def test_ahdrnet():
    model = MERGE(6, 5, 64, 32)
    x_1 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float()
    x_2 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float()
    x_3 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float()
    print(model)
    output = model(x_1, x_2, x_3)
    print(output.shape)


if __name__ == '__main__':
    test_ahdrnet()
