import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw-1.0)/2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        self.layer1 = norm_layer(
            nn.Conv2d(3, ndf, kw, stride=2, padding=pw), opt)  # 128
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw), opt)  # 64
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw), opt)  # 32
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw), opt)  # 16
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw), opt)  # 8
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw), opt)  # 4

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt

    def execute(self, x):
        # print(x)
        # print(x.size())
        if jt.size(x, 2) != 256 or jt.size(x, 3) != 256:
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = jt.view(x, (x.size(0), -1))
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
