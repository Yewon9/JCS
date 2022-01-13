import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    #x * scale + bias 리턴
    #scale = weight * running_var.rsqrt
    #bias = bias - running_mean * scale
    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    #s 리턴
    #s = weight의 열의 개수..?
    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        #use_bn=True, frozen=True이면 FrozenBatchNorm2d 실행
        #use_bn=True, frozen=False이면 BatchNorm2d 실행
        #use_bn=False이면 None
        #use_relu=True이면 nn.ReLU 실행
        #use_relu=False이면 None
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    #bn, act가 None이 아닐 경우 bn(x), act(x) 리턴
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


#conv(x) + residual_conv(x) 리턴
class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x

