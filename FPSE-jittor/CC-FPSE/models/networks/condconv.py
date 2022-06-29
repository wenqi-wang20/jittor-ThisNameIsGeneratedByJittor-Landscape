import re

import jittor as jt
import jittor.nn as nn

# depthwise seperable conv + spectral norm + batch norm


class DepthConv(nn.Module):
    def __init__(self, fmiddle, opt, kw=3, padding=1, stride=1):
        super().__init__()

        self.kw = kw
        self.stride = stride
        BNFunc = nn.BatchNorm2d

        self.norm_layer = BNFunc(fmiddle, affine=True)

    def execute(self, x, conv_weights):

        N, C, H, W = x.size()

        conv_weights = jt.view(
            conv_weights, (N * C, self.kw * self.kw, H//self.stride, W//self.stride))
        #conv_weights = nn.functional.softmax(conv_weights, dim=1)
        x = nn.unfold(x, kernel_size=(
            self.kw, self.kw), dilation=1, padding=1, stride=self.stride)
        x = jt.view(x, (N * C, self.kw * self.kw,
                        H//self.stride, W//self.stride))
        x = jt.multiply(conv_weights, x).sum(dim=1, keepdims=False)
        x = jt.view(x, (N, C, H//self.stride, W//self.stride))

        #x = self.norm_layer(x)

        return x
