'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import time
import socket
import sys
from multiprocessing import Process, Value

def rec_UDP_vm3():
    UDP_PORT = 8282
    UDP_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while(True):
        data_dummy, addr = sock.recvfrom(4096*4)
        #print(data1)
        data1.value += 1
        #print(data_dummy.value)
        #if(data_dummy.value!=0):
        #    data = data_dummy.value
        #print('received ' + str(data))
        #counter = counter + 1

def rec_UDP_vm4():
    UDP_PORT = 8181
    UDP_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while(True):
        data_dummy2, addr = sock.recvfrom(4096*4)
        #print(data1)
        data2.value += 1
        #print(data_dummy.value)
        #if(data_dummy.value!=0):
        #    data = data_dummy.value
        #print('received ' + str(data))
        #counter = counter + 1

def send_UDP_vm3():
    UDP_PORT = 8283
    UDP_IP = "10.12.0.14"
    #start = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    MESSAGE = 'X'*14
    sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
    #sock.close()
    #end = time.time()
    #print('the time is: ' + str(end-start))

def send_UDP_vm4():
    UDP_PORT = 8180
    UDP_IP = "10.12.0.19"
    #start = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    MESSAGE = 'X'*14
    sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
    #sock.close()
    #end = time.time()
    #print('the time is: ' + str(end-start))



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


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
        self.features11 = self._make_layers([64], 3)
        self.features12 = self._make_layers([64], 64)
        self.features1m = self._make_layers(['M'],64)

        self.features21 = self._make_layers([128], 64)
        self.features22 = self._make_layers([128], 128)
        self.features2m = self._make_layers(['M'], 128)

        self.features31 = self._make_layers([256], 128)
        self.features32 = self._make_layers([256], 256)
        self.features33 = self._make_layers([256], 256)
        self.features3m = self._make_layers(['M'], 256)

        self.features41 = self._make_layers([512], 256)
        self.features42 = self._make_layers([512], 512)
        self.features43 = self._make_layers([512], 512)
        self.features4m = self._make_layers(['M'], 512)

        self.features51 = self._make_layers([512], 512)
        self.features52 = self._make_layers([512], 512)
        self.features53 = self._make_layers([512], 512)
        self.features5m = self._make_layers(['M'], 512)

        #self.features3 = self._make_layers(cfg['VGG16_3'], 128)
        #self.features4 = self._make_layers(cfg['VGG16_4'], 256)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 257),
        )

    def forward(self, x):
        #p = Process(target=rec_UDP_vm3, args=())
        #p.start()
        #p1 = Process(target=rec_UDP_vm4, args=())
        #p1.start()
        #time1 = time.time()
        #out = self.features(x)
        out = self.features11(x)

        out = self.features12(out)
        out = self.features1m(out)
        out = self.features21(out)
        out = self.features22(out)
        out = self.features2m(out)

        out = self.features31(out)
        out = self.features32(out)
        out = self.features33(out)
        out = self.features3m(out)

        out = self.features41(out)
        out = self.features42(out)
        out = self.features43(out)
        out = self.features4m(out)

        out = self.features51(out)
        out = self.features52(out)
        out = self.features53(out)
        out = self.features5m(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        #f = open("machine_loss_pattern_very_heavy.txt", "a")
        #f.write("Woops! I have deleted the content!")
        #f.writelines(["%s," % item  for item in loss_list])
        #f.close()
        return out
    '''
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
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
    '''

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

def test():
    net = VGG('VGG16')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
