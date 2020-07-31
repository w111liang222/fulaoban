#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from darknet import Backbone


class Segmentator(nn.Module):
    def __init__(self, ARCH, path=None):
        super().__init__()
        self.ARCH = ARCH
        self.path = path

        # get the model
        self.backbone = Backbone(params=self.ARCH["backbone"])

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros((1,
                            self.backbone.get_input_depth(),
                            self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                            self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

        if torch.cuda.is_available():
            stub = stub.cuda()
            self.backbone.cuda()
        stub_x0 = self.backbone(stub)
        stub_x1 = self.backbone(stub)
        stub_x = torch.cat((stub_x0, stub_x1), 1)
        stub_x = stub_x.view(stub_x.size()[0], -1)

        # head
        self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                                  nn.Linear(
                                      in_features=stub_x.shape[1], out_features=6, bias=True)
                                  )

        # train backbone?
        if not self.ARCH["backbone"]["train"]:
            for w in self.backbone.parameters():
                w.requires_grad = False

        # train head?
        if not self.ARCH["head"]["train"]:
            for w in self.head.parameters():
                w.requires_grad = False

        # print number of parameters and the ones requiring gradients
        weights_total = sum(p.numel() for p in self.parameters())
        weights_grad = sum(p.numel()
                           for p in self.parameters() if p.requires_grad)
        print("Total number of parameters: ", weights_total)
        print("Total number of parameters requires_grad: ", weights_grad)

        # breakdown by layer
        weights_backbone = sum(p.numel() for p in self.backbone.parameters())
        weights_head = sum(p.numel() for p in self.head.parameters())
        print("Param backbone ", weights_backbone)
        print("Param head ", weights_head)

        # get weights
        if path is not None:
            # try backbone
            try:
                w_dict = torch.load(path + "/backbone",
                                    map_location=lambda storage, loc: storage)
                self.backbone.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model backbone weights")
            except Exception as e:
                print()
                print("Couldn't load backbone, using random weights. Error: ", e)

            # try head
            try:
                w_dict = torch.load(path + "/head",
                                    map_location=lambda storage, loc: storage)
                self.head.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)

        else:
            print("No path to pretrained, using random init.")

    def forward(self, x0, x1):
        y0 = self.backbone(x0)
        y1 = self.backbone(x1)
        y = torch.cat((y0, y1), 1)
        y = y.view(y.size()[0], -1)
        y = self.head(y)
        return y

    def save_checkpoint(self, logdir, suffix=""):
        # Save the weights
        torch.save(self.backbone.state_dict(), logdir +
                   "/backbone" + suffix)
        torch.save(self.head.state_dict(), logdir +
                   "/head" + suffix)
