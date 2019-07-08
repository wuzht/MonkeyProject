# coding: utf-8


import json
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class_path = ["n0", "n1", "n2", "n3", "n4", 
            "n5", "n6", "n7", "n8", "n9"]

base_path = {
    "train": "./10-monkey-species/training/",
    "validation": "./10-monkey-species/validation/"
}

json_path = './labels.json'


def _init_json():
    dataset = {"train":[], "validation":[]}

    # 训练集
    for i in range(len(class_path)):
        img_dir = base_path["train"] + class_path[i]
        for img_name in os.listdir(img_dir): # 对于每一种猴子
            # check the file type
            file_type = os.path.splitext(img_name)[1]
            if (not(file_type == '.jpg')):
                
                print("File is not .jpg", os.path.join(img_dir, img_name))
                continue

            dataset["train"].append({"path": os.path.join(img_dir, img_name),
                                    "label":i})
    
    # 验证集
    for i in range(len(class_path)):
        img_dir = base_path["validation"] + class_path[i]
        for img_name in os.listdir(img_dir):  # 对于每一种猴子
            # check the file type
            file_type = os.path.splitext(img_name)[1]
            if (not(file_type == '.jpg')):
                print("File is not .jpg", os.path.join(img_dir, img_name))
                continue

            dataset["validation"].append({"path": os.path.join(img_dir, img_name),
                                     "label": i})

    # 写入json
    with open(json_path, 'w') as json_file:
        json_file.write(json.dumps(dataset))


def get_mean_std():
    '''
        计算均值和方差
    '''
    number = [[],[],[]]

    if not os.path.exists(json_path):
        _init_json()

    with open(json_path) as json_file:
        dataset = json.load(json_file)['train']

    rgb_sum  = torch.zeros((3))
    rgb_squ_sum = torch.zeros((3))
    num_pixel = 0
    for i in range(len(dataset)):
        img_path = dataset[i]['path']
        img = Image.open(img_path)
        
        img = transforms.ToTensor()(img)
        # print(img_path)
        c, h, w = img.shape
        num_pixel += h*w
        # print("number of pixels:", num_pixel)

        rgb_sum += torch.sum(torch.sum(img,1),1)
        # print("total E(x):",rgb_sum)
        
        rgb_squ_sum += torch.sum(torch.sum(torch.pow(img,2),1),1)
        # print("total E(x^2):",rgb_squ_sum)

    rgb_mean = rgb_sum/num_pixel
    rgb_std = torch.sqrt(rgb_squ_sum/num_pixel - torch.pow(rgb_mean,2))

    print("mean:",rgb_mean)
    print("std:",rgb_std)

    # number = np.array(number)
    # print(number.shape)
    # print(np.mean(number[0]), np.mean(number[1]), np.mean(number[2]))
    # print(np.std(number[0]), np.std(number[1]), np.std(number[2]))

def get_shotest_size():
    '''
        获取所有图片中的最短边
    '''
    min_size = 1024 # 最短边

    if not os.path.exists(json_path):
        _init_json()

    with open(json_path) as json_file:
        dataset = json.load(json_file)['train']

    for i in range(len(dataset)):
        img_path = dataset[i]['path']
        img = Image.open(img_path)
        img = np.array(img)
        print(img_path)
        n, m, _ = img.shape
        if n < min_size: min_size = n
        if m < min_size: min_size = m
    
    print(min_size)



class MonkeyDataset(Data.Dataset):
    def __init__(self, train=True, transform=None):
        # set the paths of the images
        # assert len(imgs) == len(labels)
        if not os.path.exists(json_path):
            _init_json()

        with open(json_path) as json_file:
            if train:
                self.dataset = json.load(json_file)['train']
            else:
                self.dataset = json.load(json_file)['validation']

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.dataset[index]['path'])
        # short_size = min(np.array(img).shape[0:1])  # 得到更短边
        
        # scale and resize
        # size_transform = transforms.Compose([
        #     # transforms.CenterCrop(size=short_size), # 一些猴子不在图片中心的样本，切割后没有猴子或只有猴子的一部分
        #     transforms.Resize(size=(224,224))
        # ])
        # img = size_transform(img)
        
        # augmentation
        if self.transform != None:
            img = self.transform(img)

        label = self.dataset[index]['label']

        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    _init_json()