import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util

# Feature-Pyramid Semantics Embedding Discriminator


class FPSEDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) \
                                + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)

        # bottom-up pathway
        self.enc1 = nn.Sequential(
            norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3,
                       stride=2, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf, nf*2, kernel_size=3,
                       stride=2, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=3,
                       stride=2, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4, nf*8, kernel_size=3,
                       stride=2, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*8, nf*8, kernel_size=3,
                       stride=2, padding=1), opt),
            nn.LeakyReLU(0.2))

        # top-down pathway
        self.lat2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1), opt),
            nn.LeakyReLU(0.2))
        self.lat3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1), opt),
            nn.LeakyReLU(0.2))
        self.lat4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1), opt),
            nn.LeakyReLU(0.2))
        self.lat5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1), opt),
            nn.LeakyReLU(0.2))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.final3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2))
        self.final4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2))

        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)

    def execute(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb = self.embedding(segmap)
        segemb = nn.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb2 = nn.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = nn.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = nn.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += jt.multiply(segemb2, seg2).sum(dim=1, keepdims=True)
        pred3 += jt.multiply(segemb3, seg3).sum(dim=1, keepdims=True)
        pred4 += jt.multiply(segemb4, seg4).sum(dim=1, keepdims=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]

        return [feats, results]
