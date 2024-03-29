import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pdb
from utils import MultiHeadLinear, init_module

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out







class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.linear = nn.Linear(64, num_classes)
        self.classifier=MultiHeadLinear(64, num_classes)
        init_module(self)
        ##sampler
        self.samplers=nn.ModuleList([nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
            ) for i in range(10)]
        )
        '''
        self.sampler_fc=nn.Sequential(
                nn.Linear(64, 16, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1, bias=False),
                nn.Sigmoid()
                )
        '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def part1(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    def part2(self,x):
        out=F.avg_pool2d(x, x.size()[3])
        out=out.view(out.size(0),-1)
        out = self.classifier(out)
        return out

    def sampler(self, x, n_task):
        out=self.samplers[n_task](x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = out.mean(1)
        return out

    def getout(self, x, n_task, sampling=False):
        out1=self.part1(x)
        avg=out1.mean(0).expand_as(out1)
        out2=torch.cat((out1, avg.detach()), dim=1)
        out2=self.sampler(out2, n_task).view(-1,1,1,1)
        if sampling:
            return out2
        out2=out2*out1
        out2=self.part2(out2)
        return out2

def resnet():
    return ResNet(BasicBlock, [5, 5, 5])

