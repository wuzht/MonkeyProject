#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/05/12 02:03:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import os
os.environ['KMP_WARNINGS'] = 'off'

import torch
import torchvision.transforms as transforms
import numpy as np

# import the files of mine
from models.alexnet import *
from models.resnet import *
from models.densenet import *

import settings
from settings import log

import utility.save_load
import utility.fitting
import utility.load_dataset
from utility.load_dataset import MonkeyDataset

GPU_NOT_USE = [1, 3, 4]

# utility.load_dataset._init_json()   # print the paths and labels in json
# utility.load_dataset.get_mean_std()
# exit(-1)

# create folders
if not os.path.exists(settings.DIR_trained_model):
    os.makedirs(settings.DIR_trained_model)
if not os.path.exists(settings.DIR_logs):
    os.makedirs(settings.DIR_logs)
if not os.path.exists(settings.DIR_tblogs):
    os.makedirs(settings.DIR_tblogs)
if not os.path.exists(settings.DIR_confusions):
    os.makedirs(settings.DIR_confusions)    
if not os.path.exists(settings.DIR_confusion):
    os.makedirs(settings.DIR_confusion)

def choose_gpu():
    """
    return the id of the gpu with the most memory
    """
    # query GPU memory and save the result in `tmp`
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # read the file `tmp` to get a gpu memory list
    memory_gpu = [int(x.split()[2]) for x in open('tmp','r').readlines()]
    log.logger.info('memory_gpu: {}'.format(memory_gpu))

    for i in GPU_NOT_USE:
        memory_gpu[i] = 0   # not use these gpus

    # get the id of the gpu with the most memory
    gpu_id = str(np.argmax(memory_gpu))
    # remove the file `tmp`
    os.system('rm tmp')
    return gpu_id

device = torch.device('cuda:{}'.format(choose_gpu()))

# Hyper parameters
batch_size = 64
num_epochs = 200 # 50 100 150 200
lr = 0.0001
lr_decay_type = None
# lr_decay_type = 'linear'
num_classes = 10
momentum = 0.9
weight_decay = 4e-5


normalize = transforms.Normalize(mean=[0.4334, 0.4296, 0.3319], std=[0.2636, 0.2597, 0.2610])
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
]) if settings.isPretrain else transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
]) if settings.isPretrain else transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
train_set = MonkeyDataset(train=True, transform=train_transform)
val_set = MonkeyDataset(train=False, transform=val_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)


def get_model(_model_name, _num_classes):
    if _model_name == 'resnet34':
        return resnet34(pretrained=settings.isPretrain, num_classes=num_classes)
    elif _model_name == 'alexnet':
        return alexnet(pretrained=settings.isPretrain, num_classes=num_classes)
    elif _model_name == 'densenet121':
        return densenet121(pretrained=settings.isPretrain, num_classes=num_classes)
    else:
        log.logger.error("model_name error!")
        exit(-1)


model = get_model(settings.model_name, num_classes)
optimizer = None
if lr_decay_type == None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
loss_func = torch.nn.CrossEntropyLoss()

log.logger.critical("Preset parameters:")
log.logger.info('model_name: {}'.format(settings.model_name))
log.logger.info('isPretrain: {}'.format(settings.isPretrain))
log.logger.info('num_classes: {}'.format(num_classes))
log.logger.info('device: {}'.format(device))
log.logger.critical("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))

log.logger.critical("train_transform: \n{}".format(train_transform))
log.logger.critical("val_transform: \n{}".format(val_transform))
log.logger.critical("optimizer: \n{}".format(optimizer))
log.logger.critical("loss_func: \n{}".format(loss_func))
log.logger.critical("model: \n{}".format(model))
    
log.logger.critical('Start training')
utility.fitting.fit(model, num_epochs, optimizer, loss_func, device, train_loader, val_loader, num_classes, lr_decay_type)
log.logger.critical('Train finished')
