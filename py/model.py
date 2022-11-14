# coding=utf-8

import torch.nn as nn
import math


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, 
                     out_planes, 
                     stride=stride,
                     kernel_size=1, 
                     bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv3x3(in_planes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BottleNeck, self).__init__()
        self.conv_1 = conv1x1(in_planes, planes)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = conv3x3(planes,
                                planes,
                                stride=stride)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = conv1x1(planes, planes * 4)
        self.bn_3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(3,
                                64,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.uniform_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet50():
    model = ResNet(BottleNeck, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(BottleNeck, [3, 4, 23, 3])
    return model
