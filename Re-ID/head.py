from backbone import ResNet18Block, ResNet50Block, ResNet50_bb, ResNet18_bb
import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        pass

class MainHead(Head):
    def __init__(self):
        super(MainHead, self).__init__()
        self.in_channels = 1024
        self.layerG = self._make_layer(ResNet50Block, 512, 3, stride=2)
        self.layerL = self._make_layer(ResNet50Block, 64, 3, stride=2,local=True)

    def forward(self, x):
        G = self.layerG(x)
        G = self.avgpool(G)
        G = G.view(G.size(0), -1)

        # 1024x8x8 or 128x32x16?
        L = [None]*8
        h, w = int(x.shape[2]/4), int(x.shape[3]/2)
        k=0
        for j in range(4):
            for i in range(2):
                L[k] = x[:,:,j*h:(j+1)*h,i*w:(i+1)*w]
                L[k] = self.layerL(L[k])
                L[k] = self.avgpool(L[k])
                L[k] = L[k].view(L[k].size(0), -1)
                k+=1
        L = torch.cat((L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7]),1)

        return G, L

    def _make_layer(self, block, out_channels, num_blocks, stride, local = False):
        if local:
            self.in_channels=1024
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
        self.in_channels = 1024
        self.layerG = self._make_layer(ResNet18Block, 2048, 2, stride=1)
        self.layerL = self._make_layer(ResNet18Block, 256, 2, stride=1, local = True)

    def forward(self, L):
        G = L#torch.cat()
        #G = self.layerG(G)

        for i in range(8):
            L[i] = self.layerL(L[i])
            L[i] = self.avgpool(L[i])
            L[i] = L[i].view(L[i].size(0), -1)

        L = torch.cat((L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7]),1)
        return G,L

    def _make_layer(self, block, out_channels, num_blocks, stride, local = False):
        if local:
            self.in_channels = 128
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)



