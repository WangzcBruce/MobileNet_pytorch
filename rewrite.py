import torch
import torch.nn.functional as F
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        '''print('...')
        print(x.size())
        print(out.size())
        print('...')'''
        return out

class MobileNet(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        print(out.size())
        out = self.layers(out)
        print(out.size())
        out = F.avg_pool2d(out, 2)
        print(out.size())
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = MobileNet()
    print(net)
    input = torch.rand(1, 3, 32, 32)
    out = net(input)
    return out.size()
test()

