#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   show_curve.py
@Time    :   2019/07/07 22:14:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''


import numpy as np
import matplotlib.pyplot as plt

def show_curve_1(y1s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    plt.plot(x, y1, label='train')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_2(y1s, y2s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, label='train') # train
    plt.plot(x, y2, label='test') # test
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_3(y1s, y2s, y3s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    y3 = np.array(y3s)
    plt.plot(x, y1, label='class0')  # class0
    plt.plot(x, y2, label='class1')  # class1
    plt.plot(x, y3, label='class2')  # class2
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')