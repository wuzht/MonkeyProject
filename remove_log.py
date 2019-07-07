#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   remove_log.py
@Time    :   2019/07/07 21:45:51
@Author  :   Wu 
@Version :   1.0
@Desc    :   remove specific logs or remove all logs.
'''

import sys
import os

def remove_specific_logs(names):
    for name in names:
        log_file = os.path.join('logs', name + '.log')
        tb_dir = os.path.join('tblogs', name)
        model_file = os.path.join('trained_model', name + '.pt')
        cm_dir = os.path.join('confusions', name)
        commands = [
            'rm {}'.format(log_file),
            'rm -r {}'.format(tb_dir),
            'rm {}'.format(model_file),
            'rm -r {}'.format(cm_dir)
        ]
        for command in commands:
            try:
                print(command)
                os.system(command)
            except Exception as e:
                print(e)
                


def remove_all_logs():
    import settings
    commands = [
        'rm -r {}'.format(settings.DIR_logs),
        'rm -r {}'.format(settings.DIR_tblogs),
        'rm -r {}'.format(settings.DIR_trained_model),
        'rm -r {}'.format(settings.DIR_confusions)
    ]
    for command in commands:
        try:
            print(command)
            os.system(command)
        except Exception as e:
            print(e)
            


def main(argv):
    if len(argv) < 1:
        print('Usage: python {} all'.format(os.path.basename(__file__)))
        print('Usage: python {} <log_name1> <log_name2> ...'.format(os.path.basename(__file__)))
        exit(-1)
    print(argv)
    if argv[0] == 'all':
        remove_all_logs()
    else:
        remove_specific_logs(argv)


if __name__ == "__main__":
    main(sys.argv[1:])