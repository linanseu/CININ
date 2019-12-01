import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import Module
from filter import *
global dummy0,dummy1,dummy2,dummy3,dummy4,one,zero


'''
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
'''       



class Partition_Linear_row(nn.Linear):

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        print(self.weight[0:6,0:6])
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        #ii = torch.max(self.weight.org,0)[1]
        #bb = torch.zeros(self.weight.org.shape).cuda()
        #bb[ii,torch.linspace(0,bb.shape[1]-1,steps = bb.shape[1]).numpy()] = 1            
        self.weight.data=filter_apply(self.weight.org)    

        #print('bb shape')
        #print(bb.shape)
        #print('input shape')
        #print(input.shape)
        #print(F.linear(input, bb, bias=None))
        #torch.set_printoptions(threshold=10000)
        #print(bb)
        return F.linear(input, self.weight, bias=None)
        #return F.linear(input, bb.t(), bias=None)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Partition_Linear_col(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Partition_Linear_col, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
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
        ii = torch.max(self.weight,1)[1]
        bb = torch.zeros(self.weight.shape).cuda()
        bb[torch.linspace(0,bb.shape[1]-1,steps = bb.shape[1]).numpy(),ii] = 1
        #print('bb shape')
        #print(bb.shape)
        #print('input shape')
        #print(input.shape)
        #print(F.linear(input, bb, bias=None))
        #torch.set_printoptions(threshold=10000)
        #print(bb)
        #print(input.shape)
        #print(torch.transpose(input, 2, 1))
        return torch.transpose(F.linear(torch.transpose(input, 2, 1), bb, bias=None), 2, 1)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Lossy_Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, pieces = 4, loss_prob = 0.1, bias=True):
        super(Lossy_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.pieces = pieces
        self.block_size_x = in_features // self.pieces
        self.block_size_x_mod = in_features % self.pieces
        self.block_size_y = out_features // self.pieces
        self.block_size_y_mod = out_features % self.pieces
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
        # u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1),np.ones((self.pieces,1))),axis = 1)
        u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1), np.ones((self.pieces, self.block_size_y_mod))),axis = 1)
        mask = torch.tensor(np.repeat(u,self.block_size_x,axis = 0)).cuda()
        #print(self.weight.shape)
        #print(self.block_size_y)
        #print(self.block_size_x)
        return F.linear(input, self.weight*mask.float().t(), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Lossy_Quant_Linear(Module):
    def __init__(self, in_features, out_features, num_bits=4., lower_bound = -2., upper_bound = 2., pieces = 5, loss_prob = 0.1, bias=True):
        super(Lossy_Quant_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight0 = Parameter(torch.Tensor(out_features, in_features//pieces))
        self.weight1 = Parameter(torch.Tensor(out_features, in_features//pieces))
        self.weight2 = Parameter(torch.Tensor(out_features, in_features//pieces))
        self.weight3 = Parameter(torch.Tensor(out_features, in_features//pieces))
        self.weight4 = Parameter(torch.Tensor(out_features, in_features//pieces))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_bits = num_bits
        self.delta = (upper_bound - lower_bound) / (2 ** self.num_bits - 1)
        self.pieces = pieces
        self.block_size_x = in_features // self.pieces
        self.block_size_x_mod = in_features % self.pieces
        self.block_size_y = out_features
        self.block_size_y_mod = 0
        self.loss_prob = loss_prob
        print('aaaaaaaaaaaa')
        print(in_features//pieces)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight0.size(1))
        self.weight0.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        self.weight4.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #print('input shape is: ' + str(input.shape)) (10, 40960)
        # generate a random matrix
        #print(self.loss_prob)
        r = torch.rand((input.shape[0],self.pieces)) > self.loss_prob
        #print(r)
        # then extend it to the block random
        # u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1),np.ones((self.pieces,1))),axis = 1)
        u = torch.tensor(np.repeat(r.numpy(),self.block_size_x,axis = 1)).float().cuda()
        input = input * u
        (input0,input1,input2,input3,input4) = torch.chunk(input,5,1)
        #print('shape of input0 is ' + str(input0.shape))
        s0 = F.linear(input0, self.weight0, self.bias)
        s1 = F.linear(input1, self.weight1, self.bias)
        s2 = F.linear(input2, self.weight2, self.bias)
        s3 = F.linear(input3, self.weight3, self.bias)
        s4 = F.linear(input4, self.weight4, self.bias)
        
        s0 = F.hardtanh(s0, self.lower_bound, self.upper_bound)
        s1 = F.hardtanh(s1, self.lower_bound, self.upper_bound)
        s2 = F.hardtanh(s2, self.lower_bound, self.upper_bound)
        s3 = F.hardtanh(s3, self.lower_bound, self.upper_bound)
        s4 = F.hardtanh(s4, self.lower_bound, self.upper_bound)
        s0 = torch.round((s0-self.lower_bound) / self.delta) * self.delta + self.lower_bound
        s1 = torch.round((s1-self.lower_bound) / self.delta) * self.delta + self.lower_bound
        s2 = torch.round((s2-self.lower_bound) / self.delta) * self.delta + self.lower_bound
        s3 = torch.round((s3-self.lower_bound) / self.delta) * self.delta + self.lower_bound
        s4 = torch.round((s4-self.lower_bound) / self.delta) * self.delta + self.lower_bound
        '''
        print(torch.max(s0))
        print(torch.min(s0))
        print(torch.max(s1))
        print(torch.min(s1))
        print(torch.max(s2))
        print(torch.min(s2))
        print(torch.max(s3))
        print(torch.min(s3))
        print(torch.max(s4))
        print(torch.min(s4))
        '''
        #print('mask shape is ' + str(self.weight.shape))
        #print(self.weight.shape)
        #print(self.block_size_y)
        #print(self.block_size_x)
        return s0+s1+s2+s3+s4

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )    

    
class Masked_Linear(Module):
    def __init__(self, in_features, out_features, num_bits=4., lower_bound = -2., upper_bound = 2., pieces = 5, loss_prob = 0.1, bias=True):
        super(Masked_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.pieces = pieces
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
        #print('input shape is: ' + str(input.shape)) (10, 40960)
        # generate a random matrix
        
        # then extend it to the block random, this is when the input is 2 by 2 
        if (self.pieces == 5):
            one = torch.ones((4100//5, 40960//5))
            zero = torch.zeros((4100//5, 40960//5))
            dummy0 = torch.cat((one,zero,zero,zero,zero),dim = 1)
            dummy1 = torch.cat((zero,one,zero,zero,zero),dim = 1)
            dummy2 = torch.cat((zero,zero,one,zero,zero),dim = 1)
            dummy3 = torch.cat((zero,zero,zero,one,zero),dim = 1)
            dummy4 = torch.cat((zero,zero,zero,zero,one),dim = 1)
            dummy0 = torch.cat((dummy0,dummy1,dummy2,dummy3,dummy4),dim = 0).cuda()
        
        elif (self.pieces == 17):
        # then extend it to the block random, this is when the input is 4 by 4 
            one = torch.ones((4097//17, 34816//17))
            zero = torch.zeros((4097//17, 34816//17))
            dummy0 = torch.cat((one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy1 = torch.cat((zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy2 = torch.cat((zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy3 = torch.cat((zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy4 = torch.cat((zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy5 = torch.cat((zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy6 = torch.cat((zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy7 = torch.cat((zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy8 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy9 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy10 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero,zero),dim = 1)
            dummy11 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero,zero),dim = 1)
            dummy12 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero,zero),dim = 1)
            dummy13 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero,zero),dim = 1)
            dummy14 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero,zero),dim = 1)
            dummy15 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one,zero),dim = 1)
            dummy16 = torch.cat((zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,one),dim = 1)
            dummy0 = torch.cat((dummy0,dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7,dummy8,dummy9,dummy10,dummy11,dummy12,dummy13,dummy14,dummy15,dummy16),dim = 0).cuda()
          
        elif (self.pieces == 4):
            # then extend it to the block random, this is when the input is 2 by 2, normal partition scheme
            one = torch.ones((4096//4, 32768//4))
            zero = torch.zeros((4096//4, 32768//4))
            dummy0 = torch.cat((one,zero,zero,zero),dim = 1)
            dummy1 = torch.cat((zero,one,zero,zero),dim = 1)
            dummy2 = torch.cat((zero,zero,one,zero),dim = 1)
            dummy3 = torch.cat((zero,zero,zero,one),dim = 1)
            dummy0 = torch.cat((dummy0,dummy1,dummy2,dummy3),dim = 0).cuda()
            
        else:
            print('wrong number of pieces!')
            
        #print('weight shape is ' + str(self.weight.shape))
        #print(input.shape)
        #print(self.block_size_y)
        #print(self.block_size_x)
        return F.linear(input, self.weight*dummy0, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )        