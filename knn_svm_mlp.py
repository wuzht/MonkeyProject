#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   knn_svm_mlp.py
@Time    :   2019/07/08 04:12:49
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''


import os
import sys
import json
import torch
import torchvision.transforms as transforms
from sklearn import neighbors, svm
from sklearn.neural_network import MLPClassifier
import numpy as np

from utility.load_dataset import MonkeyDataset
from logger import ImProgressBar, Logger

import datetime

# model_name = 'knn'
model_name = 'svm'
# model_name = 'mlp'
log = Logger('./logs/{}.log'.format(model_name), level='debug')

def load_data():
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4334, 0.4296, 0.3319], std=[0.2636, 0.2597, 0.2610])
    ])
    train_set = MonkeyDataset(train=True, transform=transform)
    val_set = MonkeyDataset(train=False, transform=transform)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    print('loading training set...')
    pbar = ImProgressBar(len(train_set))
    for i, (image, label) in enumerate(train_set):
        train_images.append(image.view(-1).numpy())
        train_labels.append(label)
        pbar.update(i)
    pbar.finish()

    print('loading validation set...')
    pbar = ImProgressBar(len(val_set))
    for i, (image, label) in enumerate(val_set):
        val_images.append(image.view(-1).numpy())
        val_labels.append(label)
        pbar.update(i)
    pbar.finish()

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    print(train_images.shape, train_labels.shape)
    print(val_images.shape, val_labels.shape)
    return train_images, train_labels, val_images, val_labels



def fitting(clf, train_images, train_labels):
    print('fitting...')
    clf.fit(train_images, train_labels)
    print('done fitting')
    return clf

def evaluate(clf, val_images, val_labels, num_classes):
    print('predict...')
    y_pred = clf.predict(val_images)
    correct = np.sum(y_pred == val_labels)

    log.logger.info('y_ture: {}'.format(list(val_labels)))
    log.logger.info('y_pred: {}'.format(list(y_pred)))

    _cm = np.zeros((num_classes, num_classes + 2))
    for i, y_true in enumerate(val_labels):
        _cm[y_true, y_pred[i]] += 1
    for i in range(num_classes):
        _cm[i, -2] = np.sum(_cm[i, :num_classes])
        _cm[i, -1] = _cm[i, i] / _cm[i, -2]

    for i in range(num_classes):
        msg = ''
        for j in range(num_classes):
            msg += '{:4d}'.format(int(_cm[i, j]))
        log.logger.info('CM on validation set (class: {}) {} Acc: {}/{} ({:.2f}%)'.format(
            i,
            msg,
            int(_cm[i, i]),
            int(_cm[i, -2]),
            100 * float(_cm[i, -1])
        ))

    accuracy = correct / len(val_labels)
    log.logger.info('Accuracy on validation set: {}/{} ({:.4f}%)'.format(
        correct, len(val_labels), 100 * accuracy
    ))
    print('done predict')
    return accuracy


def perform_knn():
    best_acc = 0
    best_params = None

    train_images, train_labels, val_images, val_labels = load_data()
    for n_neighbors in range(1, 50):
        for weights in ['uniform', 'distance']:
            for metric in ['euclidean', 'manhattan', 'minkowski']:
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=8)
                log.logger.critical("model: \n{}".format(clf))
                clf = fitting(clf, train_images, train_labels)
                acc = evaluate(clf, val_images, val_labels, num_classes=10)

                if acc > best_acc:
                    best_acc = acc
                    best_params = [n_neighbors, weights, metric]
                log.logger.critical('best_acc: {:.4f}%, best_params: {}'.format(100 * best_acc, best_params))


def perform_svm():
    best_acc = 0
    best_params = None

    train_images, train_labels, val_images, val_labels = load_data()
    # kernel must be one of 'linear', 'poly', 'rbf', 'sigmoid'
    for degree in range(3, 21):
        clf = svm.SVC(gamma='scale', verbose=True, kernel='poly', degree=degree)
        log.logger.critical("model: \n{}".format(clf))
        clf = fitting(clf, train_images, train_labels)
        train_acc = evaluate(clf, train_images, train_labels, num_classes=10)
        val_acc = evaluate(clf, val_images, val_labels, num_classes=10)

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = degree
        log.logger.critical('train acc: {:.4f}%, val acc: {:.4f}%'.format(100 * train_acc, 100 * val_acc))
        log.logger.critical('best_acc: {:.4f}%, best_params: {}'.format(100 * best_acc, best_params))


def perform_mlp():
    train_images, train_labels, val_images, val_labels = load_data()
    clf = MLPClassifier(hidden_layer_sizes=(200, 200), verbose=True)
    log.logger.critical("model: \n{}".format(clf))
    clf = fitting(clf, train_images, train_labels)
    acc = evaluate(clf, val_images, val_labels, num_classes=10)


def main():
    if model_name == 'knn':
        perform_knn()
    elif model_name == 'svm':
        perform_svm()
    elif model_name == 'mlp':
        perform_mlp()
    

if __name__ == "__main__":
    main()