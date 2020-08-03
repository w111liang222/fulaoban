#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import collections
import copy
import cv2
import os
import numpy as np

from segmentator import *
from parser import Parser
from avgmeter import AverageMeter
from scipy.spatial.transform import Rotation as R


class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=self.DATA["split"]["test"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=1,
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=False)

        # concatenate the encoder and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                                     self.modeldir)

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        self.criterion = nn.SmoothL1Loss().to(self.device)

    def infer(self):
        # do train set
        # self.infer_subset(loader=self.parser.get_train_set())

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set())

        # do test set
        # self.infer_subset(loader=self.parser.get_test_set())

        print('Finished Infering')

        return

    def infer_subset(self, loader):
        losses = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        last_r = R.from_matrix([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
        last_t = np.array([0, 0, 0])
        all_pose = np.zeros((0, 12))

        rt_mat = np.hstack((last_r.as_matrix(), last_t.reshape(3, 1)))
        rt_vec = rt_mat.reshape(1, 12)
        all_pose = np.vstack((all_pose, rt_vec))

        skip_first = True
        with torch.no_grad():
            end = time.time()

            for i, (scan0, scan1, delta_pose) in enumerate(loader):
                if skip_first:
                    skip_first = False
                    continue
                if self.gpu:
                    scan0 = scan0.cuda()
                    scan1 = scan1.cuda()
                    delta_pose = delta_pose.cuda(non_blocking=True).float()
                # compute output
                output = self.model(scan0, scan1)

                # calculate loss
                loss_t = self.criterion(output[:, 0: 3], delta_pose[:, 0: 3])
                loss_r = self.criterion(output[:, 3:], delta_pose[:, 3:])
                loss = loss_t + loss_r
                loss = loss.mean()
                losses.update(loss)

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                print("Infered seq scan ", i,
                      " in", time.time() - end, "sec", "lose", losses.avg.cpu().numpy())
                end = time.time()

                output_np = output.cpu().numpy().reshape(6,)
                output_t = output_np[0:3]
                output_r = R.from_euler('zxy', output_np[3:], degrees=True)

                last_t = last_t + np.dot(last_r.as_matrix(), output_t)
                last_r = last_r*output_r

                rt_mat = np.hstack((last_r.as_matrix(), last_t.reshape(3, 1)))
                rt_vec = rt_mat.reshape(1, 12)
                all_pose = np.vstack((all_pose, rt_vec))
                # if i > 300:
                #    break
            # save scan
            path = os.path.join(self.logdir, 'pose.txt')
            np.savetxt(path, all_pose, delimiter=' ')
