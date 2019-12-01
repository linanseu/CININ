import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
from input_activation import Lossy_Linear, Lossy_Quant_Linear, Masked_Linear
from markov_random import markov_rand
import time
global nonzero_pixels_rate
global bytes_per_packet
nonzero_pixels = []
bytes_per_packet = []

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_add_middle(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # has to first partition the things into halves
    (x1,x2) = torch.chunk(x,2,2)
    (x11,x12) = torch.chunk(x1,2,3)
    (x21,x22) = torch.chunk(x2,2,3)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class lossy_Conv2d_new(nn.Module):   
    def __init__(self, in_channels, out_channels, p11 = 0.99, p22 = 0.03, kernel_size=3, stride = 0, padding=0, num_pieces=(2, 2), bias=False):
        super(lossy_Conv2d_new, self).__init__()
        # for each pieces, define a new conv operation
        self.pieces = num_pieces
        self.p11 = p11
        self.p22 = p22
        self.stride = stride
        self.b1 = nn.Sequential(
            # use the parameters instead of numbers
            nn.Conv2d(in_channels, out_channels, stride = self.stride, kernel_size=kernel_size, padding=0, bias=False)
        )
    def forward(self, x):
        # print("x shape : ", x.shape)
        def split_dropout(x, pieces):
            
            dim = x.shape
            l_i = dim[2] // pieces[0]
            l_j = dim[3] // pieces[1]
            
            x_split = []
            for i in range(pieces[0]):
                dummy = []
                for j in range(pieces[1]):
                    x_s = 0 if i == 0 else i * l_i - 1
                    y_s = 0 if j == 0 else j * l_j - 1
                    x_e = (i + 1) * l_i if i == pieces[0] - 1 else (i + 1) * l_i + 1
                    y_e = (j + 1) * l_j if j == pieces[1] - 1 else (j + 1) * l_j + 1
                    xx = x[:, :, x_s: x_e, y_s: y_e]
                    xx = F.pad(xx, (int(j == 0), int(j == pieces[1] - 1), int(i == 0), int(i == pieces[0] - 1), 0, 0, 0, 0))
                    xx = xx.cuda()
                    mask = markov_rand(xx.shape, self.p11, self.p22)
                    xx = xx * mask.cuda()                    
                    xx[:, :, 1: 1 + l_i, 1: 1 + l_j] = x[:, :, i * l_i: (i + 1) * l_i, j * l_j: (j + 1) * l_j]
                    dummy.append(xx.cuda())
                x_split.append(dummy)
            return x_split
        x_split = split_dropout(x, self.pieces)
        r = []
        for i in range(self.pieces[0]):
            dummy = []
            for j in range(self.pieces[1]):
                rr = self.b1(x_split[i][j])
                dummy.append(rr)
            dummy_cat = torch.cat((dummy[0: self.pieces[1]]), 3)
            r.append(dummy_cat)    
        r = torch.cat((r[0: self.pieces[0]]), 2)
        return r.cuda()

class Quant_ReLU(nn.Module):
    def __init__(self, lower_bound=0.8, upper_bound=1., num_bits=4., num_pieces=(2, 2)):
        super(Quant_ReLU, self).__init__()
        # for each pieces, define a new conv operation
        self.num_bits = num_bits
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.delta = (upper_bound - lower_bound) / (2 ** self.num_bits - 1)
        self.num_pieces = num_pieces

    def forward(self, x):
        def gen_mask(dim=(16, 16, 4, 4), pieces=(2, 2)):
            mask = torch.zeros(dim[2], dim[3])
            for i in range(1, pieces[0]):
                mask[i * dim[2] // pieces[0] - 1, :] = 1;
                mask[i * dim[2] // pieces[0], :] = 1;
            for j in range(1, pieces[1]):
                mask[:, j * dim[3] // pieces[1] - 1] = 1;
                mask[:, j * dim[3] // pieces[1]] = 1;
            return mask.cuda()

        mask = gen_mask(x.shape, self.num_pieces)
        r1 = F.hardtanh(x * mask, self.lower_bound, self.upper_bound) - self.lower_bound
        r = F.relu(x * (1 - mask)) + r1 * mask
        return r

    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class Bottleneck_lossy(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, lower_bound = 2, upper_bound = 3, num_bits = 4., p11 = 0.99, p22 = 0.03, num_pieces = (2,2), downsample=None, add_middle_reslink = 0):
        super(Bottleneck_lossy, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.quant_relu = Quant_ReLU(lower_bound=lower_bound, upper_bound=upper_bound, num_bits=num_bits)
        self.downsample = downsample
        self.stride = stride
        self.add_middle_reslink = add_middle_reslink
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.quant_relu(out)

        out = self.conv2(out)   # the strided convolution is here
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if (self.add_middle_reslink == 0):
                identity = self.downsample(x)
            elif (self.add_middle_reslink == 1):
                (x1,x2) = torch.chunk(x,2,2)
                (x11,x12) = torch.chunk(x1,2,3)
                (x21,x22) = torch.chunk(x2,2,3)
                x11 = self.downsample(x11)
                x12 = self.downsample(x12)
                x21 = self.downsample(x21)
                x22 = self.downsample(x22)
                x1 = torch.cat((x11,x12),3)
                x2 = torch.cat((x21,x22),3)    
                identity = torch.cat((x1,x2),2) 
        out += identity
        out = self.relu(out)
        
        return out
    
    
class ResNet(nn.Module):

    def __init__(self, block, block_lossy, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        #################################
        self.p11 = 0.99
        self.p22 = 0.03
        if (num_classes == 257):
            self.lower_bound = 0.55
            self.upper_bound = 0.6
        if (num_classes == 101):    
            self.lower_bound = 1.0
            self.upper_bound = 2.25
        print(self.lower_bound)
        print(self.upper_bound)
        self.num_bits = 4.
        self.f12_pieces = (2,2)
        ################################# 
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_lossy_conv(block_lossy, 64, layers[0], stride=1, p11 = self.p11, p22 = self.p22, lower_bound = self.lower_bound, upper_bound = self.upper_bound, num_bits = self.num_bits, pieces = self.f12_pieces)
        self.layer2 = self._make_layer_lossy_conv(block_lossy, 128, layers[1], stride=2, p11 = self.p11, p22 = self.p22, lower_bound = self.lower_bound, upper_bound = self.upper_bound, num_bits = self.num_bits, pieces = self.f12_pieces)
        self.layer3 = self._make_layer_lossy_conv(block_lossy, 256, layers[2], stride=2, p11 = self.p11, p22 = self.p22, lower_bound = self.lower_bound, upper_bound = self.upper_bound, num_bits = self.num_bits, pieces = self.f12_pieces)
        self.layer4 = self._make_layer_lossy_conv(block_lossy, 512, layers[3], stride=2, p11 = self.p11, p22 = self.p22, lower_bound = self.lower_bound, upper_bound = self.upper_bound, num_bits = self.num_bits, pieces = self.f12_pieces, add_middle_reslink = 1)    # the reslink for this layer, the 1 by 1 conv with stride 2 has to be implemented in a special way
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_lossy_conv(self, block_lossy, planes, blocks, stride=1, p11 = 0.99, p22 = 0.03, lower_bound = 2, upper_bound = 3, num_bits = 4., pieces = (2,2), add_middle_reslink = 0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_lossy.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block_lossy.expansion, stride),
                nn.BatchNorm2d(planes * block_lossy.expansion),
            )
        layers = []
        layers.append(block_lossy(self.inplanes, planes, stride, lower_bound = lower_bound, upper_bound = upper_bound, num_bits = num_bits, downsample = downsample, add_middle_reslink = add_middle_reslink))
        self.inplanes = planes * block_lossy.expansion
        for _ in range(1, blocks):
            layers.append(block_lossy(self.inplanes, planes, lower_bound = lower_bound, upper_bound = upper_bound, num_bits = num_bits))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        (x1,x2) = torch.chunk(x,2,2)
        (x11,x12) = torch.chunk(x1,2,3)
        (x21,x22) = torch.chunk(x2,2,3)
        x11 = self.maxpool(self.relu(self.bn1(self.conv1(x11))))
        x12 = self.maxpool(self.relu(self.bn1(self.conv1(x12))))
        x21 = self.maxpool(self.relu(self.bn1(self.conv1(x21))))
        x22 = self.maxpool(self.relu(self.bn1(self.conv1(x22))))
        x1 = torch.cat((x11,x12),3)
        x2 = torch.cat((x21,x22),3)    
        x = torch.cat((x1,x2),2) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, num_classes = 101):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, num_classes = 101):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes = 101)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, num_classes = 257):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, Bottleneck_lossy, [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model