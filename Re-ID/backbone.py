import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.io import read_image
import torchvision.transforms as T

class ResNet50Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet50Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet50_bb(nn.Module):
    def __init__(self):
        super(ResNet50_bb, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNet50Block, 64, 3, stride=1)
        #self.layer2 = self._make_layer(ResNet50Block, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResNet50Block, 256, 6, stride=2)
        #self.layer4 = self._make_layer(ResNet50Block, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 2048)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer3(out)
        print(out.shape)
        # out = self.avgpool(out)
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        # out = self.fc(out)
        # print(out.shape)
        return out

class ResNet18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet18Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        
        return out

class ResNet18_bb(nn.Module):
    def __init__(self):
        super(ResNet18_bb, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNet18Block, 64, 2, stride=2)
        self.layer2 = self._make_layer(ResNet18Block, 128, 2, stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(1024, 256)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Run part 1 for 24 DP-images
        outs = [None in range(24)]
        for i in range(24):
            outs[i] = self.conv1(x[i])
            outs[i] = self.bn1(outs[i])
            outs[i] = self.relu(outs[i])
            outs[i] = self.conv2(outs[i])
            outs[i] = self.relu(outs[i])
            outs[i] = self.layer1(outs[i])

        # Merge round 1
        outs = merge1(outs)

        # Run part 2 for 13 DP-images
        for i in range(13):
            outs[i] = self.layer2(outs[i])

        # Merge round 2
        outs = merge2(outs)
        print(outs.shape)
        return outs

def merge1(X):
     merged_X = [None in range(13)]
     merged_X[0], merged_X[1] = X[0], X[1]
     i=2
     for k in range(2,24,2):
         merged_X[i] = X[k] + X[k+1]
         i += 1
     return merged_X

def merge2(X):
    merged_X = [None in range(8)]
    merged_X[0] = X[0] + X[1]
    merged_X[1] = X[2]
    merged_X[2] = X[3]
    merged_X[3] = X[4] + X[5]
    merged_X[4] = X[6] + X[7]
    merged_X[5] = X[8] + X[9]
    merged_X[6] = X[10] + X[11]
    merged_X[7] = X[12]
    return merged_X

if __name__ == '__main__':
    # model = ResNet18_bb().to('cuda')
    # im_path = '/mnt/analyticsvideo/DensePoseData/market1501/SegmentedMarket1501train/0002/uv_maps'
    # im = read_image(im_path + '/0002_c1s1_000451_03_texture.jpg')
    # x = [None] * 24
    # k = 0
    # for i in range(4):
    #     for j in range(6):
    #         x[k] = im[:,i*200:(i+1)*200,j*200:(j+1)*200].to(torch.float).to('cuda')
    #         k+=1   

    # x = model(x).to('cuda')
    pass
