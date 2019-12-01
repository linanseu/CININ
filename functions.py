b# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:30:49 2018

@author: akash
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# for col
class BinarizeFF(Function):

    @staticmethod
    def forward(cxt, input):
        #output = input.new(input.size())
        output = torch.zeros(input.size()).cuda()
        #print(input)
        ii = torch.max(input,1)[1]
        #print(ii)
        #print(torch.linspace(0,input.shape[1]-1,steps = input.shape[1]).numpy())
        #print(torch.linspace(0,input.shape[0]-1,steps = input.shape[0]).numpy()).shape
        #bb = torch.zeros(self.weight.shape).cuda()
        output[torch.linspace(0,input.shape[1]-1,steps = input.shape[1]).numpy(),ii] = 1
        #output[input >= 0] = 1
        #output[input < 0] = -1
        #print(output)     
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize1 = BinarizeFF.apply

# for row
class BinarizeFFF(Function):

    @staticmethod
    def forward(cxt, input):
        #print('data receive')
        #print(input)
        #output = input.new(input.size())
        output = torch.zeros(input.size()).cuda()
        ii = torch.max(input,0)[1]
        output[ii,torch.linspace(0,input.shape[1]-1,steps = input.shape[1]).numpy()] = 1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize2 = BinarizeFFF.apply

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

class TernarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0.1] = 1
        output[input <= -0.1] = -1
        #print(output.view(1,-1))
        #output[input>= -0.5 and input <= 0.5] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
ternarize = TernarizeF.apply