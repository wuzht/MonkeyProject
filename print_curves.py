#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   print_curves.py
@Time    :   2019/07/07 21:45:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   get accuracy and loss from log file, then print curves
'''

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

def show_curve_2(y1s, y2s, title, ylabel, isAcc=True):
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, label='Training') # train
    plt.plot(x, y2, label='Validation') # test
    plt.axis()
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(ylabel))
    
    if isAcc is True:
        plt.yticks(np.linspace(0, 1, 11))

    plt.legend(loc='best')
    plt.grid()
    plt.show()
    # plt.savefig("{}.svg".format(ylabel))
    plt.close()

def get_data_from_log(name):
    """
    get accuracy and loss from log file
    """
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    with open('logs/{}.log'.format(name), 'r') as f:
        for line in f.readlines():
            line = line.strip()

            train = re.findall(r'Train.*Acc:.*\((.*)\%\)', line)
            if len(train) > 0:
                train_accs.append(float(train[0]) / 100)

            val = re.findall(r'Val.*Acc:.*\((.*)\%\)', line)
            if len(val) > 0:
                val_accs.append(float(val[0]) / 100)

            train_loss = re.findall(r'Train.*Avg loss: (.*), Acc:', line)
            if len(train_loss) > 0:
                train_losses.append(float(train_loss[0]))
            
            val_loss = re.findall(r'Val.*Avg loss: (.*), Acc:', line)
            if len(val_loss) > 0:
                val_losses.append(float(val_loss[0]))
    return train_accs, val_accs, train_losses, val_losses


def main(argv):
    print(argv)
    train_accs, val_accs, train_losses, val_losses = get_data_from_log(argv[0])
    show_curve_2(train_accs, val_accs, 'Accuracy', 'Accuracy', True)
    show_curve_2(train_losses, val_losses, 'Loss', 'Loss', False)

if __name__ == "__main__":
    main(sys.argv[1:])
