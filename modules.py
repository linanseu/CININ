# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:29:33 2018

@author: akash
"""

import math
import numpy as np
#import torch.nn.parameter as Parameter
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from functions import *
from markov_random import markov_rand


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output

class TernarizeTanh(nn.Module):
    def __init__(self):
        super(TernarizeTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = ternarize(output)
        return output

class BinaryLinear(nn.Linear):

    def forward(self, input):
        #binary_weight = binarize(self.weight)
        if input.size(1) != 784:
            input.data=binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

class TenLinear(nn.Linear):
    
    def forward(self, input):
        #binary_weight = binarize(self.weight)
        if input.size(1) != 784:
            input.data=ternarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=ternarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

class Col_exchange(nn.Linear):
    def forward(self, input):
        #if input.size(1) != 784:
        #    input.data=binarize1(input.data)
        if not hasattr(self.weight,'org'):
            #print('yes1')
            self.weight.org=self.weight.data.clone()
        #print('nenenene')
        #print(self.weight[0:5,0:5])         
        self.weight.data=binarize2(self.weight.org)
        #print(self.weight[0:5,0:5])         
        out = nn.functional.linear(input, self.weight.t())

        return out

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

class Row_exchange(nn.Linear):
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            #print('yes1')
            self.weight.org=self.weight.data.clone()
        #print('nenenene')
        #print(self.weight[0:5,0:5])         
        self.weight.data=binarize1(self.weight.org)
        #print(self.weight[0:5,0:5])         
        out = nn.functional.linear(self.weight, input.t())

        return out

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

class Lossy_Linear(Module):

    def __init__(self, in_features, out_features, pieces = 4, loss_prob = 0.1, bias=True):
        super(Lossy_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.pieces = pieces
        self.block_size_x = in_features/self.pieces
        self.block_size_y = out_features/self.pieces
        self.loss_prob = loss_prob
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # generate a random matrix
        r = torch.rand((self.pieces,self.pieces)) > self.loss_prob
        # then extend it to the block random
        u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1),np.ones((self.pieces,1))),axis = 1)
        mask = torch.tensor(np.repeat(u,self.block_size_x,axis = 0)).float().cuda()
        return F.linear(input, self.weight*mask.float().t()[0:self.weight.shape[0],:], self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TenConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TenConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = ternarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=ternarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class BinConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        #self.mask = mask
        super(BinConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        #self.weight.data=self.weight.data * mask
        #print(self.weight.data)
        self.weight.data=binarize(self.weight.org)
        
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out