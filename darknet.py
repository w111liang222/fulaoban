# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.layers = params["extra"]["layers"]
        print("Using DarknetNet" + str(self.layers) + " Backbone")

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        print("Depth of backbone input = ", self.input_depth)

        # stride play
        self.strides = [2, 2, 2, 2, 2]

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                         stride=self.strides[0], bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                         stride=self.strides[1], bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2], bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3], bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                         stride=self.strides[4], bn_d=self.bn_d)

        # ts net
        self.dec5 = self._make_dec_layer([1024, 256],
                                         bn_d=self.bn_d,
                                         stride=self.strides[4])
        self.dec5_conv = self._make_enc_layer(None, [768, 256], 0,
                                              stride=1, bn_d=self.bn_d, use_res=False)
        self.dec4 = self._make_dec_layer([256, 128], bn_d=self.bn_d,
                                         stride=self.strides[3])
        self.dec4_conv1 = self._make_enc_layer(None, [384, 128], 0,
                                               stride=1, bn_d=self.bn_d, use_res=False)
        self.dec4_conv2 = self._make_enc_layer(None, [128, 32], 0,
                                               stride=1, bn_d=self.bn_d, use_res=False)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 32

    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1, use_res=True):
        layers = []

        #  downsample
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=[1, stride], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        if use_res:
            inplanes = planes[1]
            for i in range(0, blocks):
                layers.append(("residual_{}".format(i),
                               block(inplanes, planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def _make_dec_layer(self, planes, bn_d=0.1, stride=2):
        layers = []

        #  upsample
        layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                    kernel_size=[1, 4], stride=[1, 2],
                                                    padding=[0, 1])))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skip=False):
        y = layer(x)
        if skip:
            x_skip = x.detach()
        else:
            x_skip = None
        x = y
        return x, x_skip

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # first layer
        x, _ = self.run_layer(x, self.conv1)
        x, _ = self.run_layer(x, self.bn1)
        x, _ = self.run_layer(x, self.relu1)

        # all encoder blocks with intermediate dropouts
        x, _ = self.run_layer(x, self.enc1)
        x, _ = self.run_layer(x, self.dropout)
        x, _ = self.run_layer(x, self.enc2)
        x, _ = self.run_layer(x, self.dropout)
        x, _ = self.run_layer(x, self.enc3)
        x, x_skip1 = self.run_layer(x, self.dropout, skip=True)
        x, _ = self.run_layer(x, self.enc4)
        x, x_skip2 = self.run_layer(x, self.dropout, skip=True)
        x, _ = self.run_layer(x, self.enc5)
        x, _ = self.run_layer(x, self.dropout)

        # ts-net
        x, _ = self.run_layer(x, self.dec5)
        x = torch.cat((x_skip2, x), 1)
        x, _ = self.run_layer(x, self.dec5_conv)
        x, _ = self.run_layer(x, self.dec4)
        x = torch.cat((x_skip1, x), 1)
        x, _ = self.run_layer(x, self.dec4_conv1)
        x, _ = self.run_layer(x, self.dec4_conv2)

        return x

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
