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
from matplotlib import pyplot as plt
from parser import Parser
from logger import Logger
from avgmeter import AverageMeter
from segmentator import Segmentator
from warmupLR import warmupLR
from sync_batchnorm.batchnorm import convert_model


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        # put logger where it belongs
        self.tb_logger = Logger(self.log + "/tb")
        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "valid_loss": 0,
                     "backbone_lr": 0,
                     "head_lr": 0}

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=None,
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.ARCH["train"]["batch_size"],
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=True)

        # concatenate the backbone and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                                     self.path)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = False
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)   # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        # loss
        self.criterion = nn.SmoothL1Loss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(
                self.criterion).cuda()  # spread in gpus

        # optimizer
        self.lr_group_names = []
        self.train_dicts = []
        if self.ARCH["backbone"]["train"]:
            self.lr_group_names.append("backbone_lr")
            self.train_dicts.append(
                {'params': self.model_single.backbone.parameters()})
        if self.ARCH["head"]["train"]:
            self.lr_group_names.append("head_lr")
            self.train_dicts.append(
                {'params': self.model_single.head.parameters()})

        # Use SGD optimizer to train
        self.optimizer = optim.SGD(self.train_dicts,
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    def train(self):
        best_val_loss = 1e10
        # train for n epochs
        for epoch in range(self.ARCH["train"]["max_epochs"]):
            # get info for learn rate currently
            groups = self.optimizer.param_groups
            for name, g in zip(self.lr_group_names, groups):
                self.info[name] = g['lr']

            # train for 1 epoch
            loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                 model=self.model,
                                                 criterion=self.criterion,
                                                 optimizer=self.optimizer,
                                                 epoch=epoch,
                                                 scheduler=self.scheduler,
                                                 report=self.ARCH["train"]["report_batch"])

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss

            self.model_single.save_checkpoint(self.log, suffix="_train")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                loss = self.validate(val_loader=self.parser.get_valid_set(),
                                     model=self.model,
                                     criterion=self.criterion)

                # update info
                self.info["valid_loss"] = loss

                if best_val_loss > loss:
                    best_val_loss = loss
                    # save the weights!
                    print("best mean loss in validation so far, save model!")
                    self.model_single.save_checkpoint(self.log, suffix="")

                print("*" * 80)

                # save to log
                Trainer.save_to_log(logdir=self.log,
                                    logger=self.tb_logger,
                                    info=self.info,
                                    epoch=epoch,
                                    w_summary=self.ARCH["train"]["save_summary"],
                                    model=self.model_single)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, scheduler, report=10):
        batch_time = AverageMeter()
        losses = AverageMeter()
        update_ratio_meter = AverageMeter()

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (scan0, scan1, delta_pose) in enumerate(train_loader):
            if not self.multi_gpu and self.gpu:
                scan0 = scan0.cuda()
                scan1 = scan1.cuda()
                delta_pose = delta_pose.cuda()
            if self.gpu:
                delta_pose = delta_pose.cuda(non_blocking=True).float()

            # compute output
            output = model(scan0, scan1)
            loss = criterion(output, delta_pose)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss.backward(idx)
            else:
                loss.backward()

            optimizer.step()

            # measure accuracy and record loss
            loss = loss.mean()
            losses.update(loss)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(
                            value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if i % self.ARCH["train"]["report_batch"] == 0:
                print('Lr: {lr:.3e} | '
                      'Update: {umean:.3e} mean,{ustd:.3e} std | '
                      'Epoch: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          loss=losses, lr=lr,
                          umean=update_mean, ustd=update_std))

            # step scheduler
            scheduler.step()

        return losses.avg, update_ratio_meter.avg

    def validate(self, val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (scan0, scan1, delta_pose) in enumerate(val_loader):
                if not self.multi_gpu and self.gpu:
                    scan0 = scan0.cuda()
                    scan1 = scan1.cuda()
                    delta_pose = delta_pose.cuda()
                if self.gpu:
                    delta_pose = delta_pose.cuda(non_blocking=True).float()

                # compute output
                output = model(scan0, scan1)
                loss = criterion(output, delta_pose)

                # record loss
                loss = loss.mean()
                losses.update(loss)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            print('Validation set:\n'
                  'Time avg per batch {batch_time.avg:.3f}\n'
                  'Loss avg {loss.avg:.4f}\n'.format(batch_time=batch_time,
                                                     loss=losses,
                                                     ))

        return losses.avg
