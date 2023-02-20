from backbone import ResNet18Block, ResNet50Block, ResNet50_bb, ResNet18_bb
import torch.nn as nn

class Head(nn.module):
    def __init__(self):
        super(Head, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        pass

class MainHead(Head):
    def __init__(self):
        super(MainHead, self).__init__()
        self.layer1 = self._make_layer(ResNet50Block, 512, 3, stride=1)

    def forward(self, x):
        pass

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)


class DenseHead(Head):
    def __init__(self):
        super(DenseHead,self).__init__()
        self.layer1 = self.make_layer(ResNet18Block, 64, 2, stride=2)

    def forward(self, x):
        pass

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)



