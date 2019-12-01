# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:30:49 2018

@author: akash
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class BinarizeFF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        #print(output.shape)
        ii = torch.max(input,1)[1]
        #print(ii.shape)
        #print(torch.linspace(0,input.shape[0]-1,steps = input.shape[0]).numpy()).shape
        #bb = torch.zeros(self.weight.shape).cuda()
        output[torch.linspace(0,input.shape[0]-1,steps = input.shape[0]).numpy(),ii] = 1
        #output[input >= 0] = 1
        #output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize1 = BinarizeFF.apply

class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply
