#coding=utf-8

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
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

num_gpu = 2
#nonzero_pixels_rate = np.zeros((6, num_gpu), dtype=float)
#bytes_per_packet = np.zeros((6, num_gpu), dtype=float)
print_flag = False

def set_print_flag(flag):
    global print_flag
    print_flag = flag

def init_array():
    global nonzero_pixels_rate
    global bytes_per_packet
    nonzero_pixels_rate = np.zeros((6, num_gpu), dtype=float)
    bytes_per_packet = np.zeros((6, num_gpu), dtype=float)
    
def get_array():
    nonzero = nonzero_pixels_rate.mean(axis=1)
    bytes_per = bytes_per_packet.mean(axis=1)
    return nonzero, bytes_per


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_1': [64, 64, 'M'],
    'VGG16_2': [128, 128, 'M'],
    'VGG16_3': [256, 256, 256, 'M'],
    'VGG16_4': [512, 512, 512, 'M'],
    'VGG16_5': [512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}




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
        
        ################################################################
        # quantize the pixels on the margin
        r1 = torch.round(r1 / self.delta) * self.delta
        # applies different mask to the pixel in the middle and on the margin
        r = F.relu(x * (1 - mask)) + r1
        return r


class VGG(nn.Module):
    def __init__(self, vgg_name, dataset, original, p11=0.99, p22=0.03, lossyLinear=False, loss_prob=0.1, pieces=(2, 2), f12_pieces=(2, 2), lower_bound=0.5, upper_bound=1.0):
        super(VGG, self).__init__()
        # Hyperparameters
        #################################
        self.loss_prob = 0.01
        self.lower_bound = 0.1
        self.upper_bound = 1.2
        self.num_bits = 4
        self.pieces = 4
        self.f12_pieces = (2,2)
        #################################
        self.features1 = self._make_layers(cfg['VGG16_1'], 3)
        self.features2 = self._make_layers(cfg['VGG16_2'], 64)
        self.original = original
        if original:
            self.features3 = self._make_layers(cfg['VGG16_3'], 128)
            self.features4 = self._make_layers(cfg['VGG16_4'], 256)
            self.features5 = self._make_layers(cfg['VGG16_5'], 512)
        else :
            self.features3 = self._make_layers_lossy_conv(cfg['VGG16_3'], 128, p11, p22, dataset, pieces, lower_bound=lower_bound, upper_bound=upper_bound)
            self.features4 = self._make_layers_lossy_conv(cfg['VGG16_4'], 256, p11, p22, dataset, pieces, lower_bound=lower_bound, upper_bound=upper_bound)
            self.features5 = self._make_layers_lossy_conv_layer5(cfg['VGG16_5'], 512, p11, p22, dataset, pieces, lower_bound=lower_bound, upper_bound=upper_bound)
        if dataset == 'Cifar10':
            if lossyLinear:
                self.classifier = Lossy_Linear(512, 10, loss_prob=loss_prob)
            else:
                self.classifier = nn.Linear(512, 10)
        elif dataset == 'Caltech256':
                self.classifier = nn.Sequential(
                    Lossy_Quant_Linear(40960, 4096, pieces = self.pieces, num_bits = self.num_bits, lower_bound = self.lower_bound, upper_bound = self.upper_bound, loss_prob = self.loss_prob, bias=True),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 257)
                )
        elif dataset == 'Caltech101':
                self.classifier = nn.Sequential(
                    Masked_Linear(32768, 4096, pieces = self.pieces, loss_prob = self.loss_prob, bias=True),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 101)
                )

    def forward(self, x):
        if self.original:
            out = self.features1(x)
            out = self.features2(out)
        else:
            x_split = []
            xx = torch.chunk(x, self.f12_pieces[0], 2)
            for i in range(self.f12_pieces[0]):
                xxx = torch.chunk(xx[i], self.f12_pieces[1], 3)
                x_split.append(xxx)
        
            out = []
            for i in range(self.f12_pieces[0]):
                dummy = []
                for j in range(self.f12_pieces[1]):
                    rr = self.features1(x_split[i][j].cuda())
                    rr = self.features2(rr.cuda())
                    dummy.append(rr)
                dummy_cat = torch.cat((dummy[0: self.f12_pieces[1]]), 3)
                out.append(dummy_cat)    
            out = torch.cat((out[0: self.f12_pieces[0]]), 2)
            out.cuda()
        out = self.features3(out)
        out = self.features4(out)
        
        out = self.features5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels, relu_change=0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _make_layers_lossy_conv(self, cfg, in_channels, p11, p22, dataset, pieces=(2, 2), relu_change=0, lower_bound=0.5, upper_bound=1.0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [lossy_Conv2d_new(in_channels, x, stride = 1, kernel_size=3, p11 = p11, p22 = p22, num_pieces=pieces),
                           nn.BatchNorm2d(x, affine=False),
                           Quant_ReLU(lower_bound=lower_bound, upper_bound=upper_bound, num_bits=4., num_pieces=pieces)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _make_layers_lossy_conv_layer5(self, cfg, in_channels, p11, p22, dataset, pieces=(2, 2), relu_change=0, lower_bound=0.5, upper_bound=1.0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)]   # padding = 1 is for the splitting the max pooling layer for the 2 by 2 partition
            else:
                layers += [lossy_Conv2d_new(in_channels, x, stride = 1, kernel_size=3, p11 = p11, p22 = p22, num_pieces=pieces),
                           nn.BatchNorm2d(x, affine=False),
                           nn.ReLU(True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG16', 'Caltech101', False, lossyLinear=True, lower_bound=1.999, upper_bound=2.0, pieces=(2, 2), f12_pieces=(2, 2))
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    x = torch.randn(256, 3, 224, 224)
    # x = torch.randn(128, 3, 32, 32)
    init_array()
    y = net(x)
    # print_array()
    
