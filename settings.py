#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   py
@Time    :   2019/05/12 01:48:31
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import os
import datetime

# import the files of mine
import logger

## setttings #############################################
# model_name = 'resnet34'
# model_name = 'densenet121'
# model_name = 'alexnet'
# model_name = 'vgg16_bn'
model_name = 'shufflenetv2_x1_0'

isPretrain = True

# folders
DIR_trained_model = './trained_model/'
DIR_logs = './logs/'
DIR_tblogs = './tblogs/'
DIR_confusions = './confusions/'

##########################################################
now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S') # now_time: to name the following variables
name = '{}={}'.format(now_time, model_name)
PATH_model = os.path.join(DIR_trained_model, '{}.pt'.format(name))
DIR_tblog = os.path.join(DIR_tblogs, name)
PATH_log = os.path.join(DIR_logs, '{}.log'.format(name))
DIR_confusion = os.path.join(DIR_confusions, name)
DIR_tb_cm  = os.path.join(DIR_tblog, 'cm')  # confusion matrix

# create folders
if not os.path.exists(DIR_trained_model):
    os.makedirs(DIR_trained_model)
if not os.path.exists(DIR_logs):
    os.makedirs(DIR_logs)
if not os.path.exists(DIR_tblogs):
    os.makedirs(DIR_tblogs)
if not os.path.exists(DIR_confusions):
    os.makedirs(DIR_confusions)    
if not os.path.exists(DIR_confusion):
    os.makedirs(DIR_confusion)

log = logger.Logger(PATH_log, level='debug')