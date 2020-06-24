# import matplotlib as mpl
# import matplotlib.pylab as plt
# import pandas as pd 
# from numpy.random import *
# import numpy as np 

# mean = [0, 0, 0]
# cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# dataset = np.array(multivariate_normal(mean, cov, 5000))
# np.savetxt('test.txt', dataset)
# dataset = pd.DataFrame(multivariate_normal(mean, cov, 5000))
# dataset.to_csv("test.csv", sep=" ")

# fig=plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal', adjustable='box')
# H = ax.hist2d(dataset["Birth"], dataset["Death"],bins=128,
#               norm=mpl.colors.LogNorm(), cmap="rainbow")
# L = ax.plot([0,1], [0,1], c="black", lw=1)

# ax.set_xlabel('Birth')
# ax.set_ylabel('Death')
# fig.colorbar(H[3],ax=ax)

import csv
import glob
from enum import Enum
import sys
from voice_state import VoiceState
from voice_feature import VoiceFeacure
from distutils.util import strtobool
import os

class FeatureValue :
    def __init__(self, is_labeled, target_dir, question_path) :
        self.__QESTION_FILEPATH = question_path
        self.__VAD_DIRPATH = ''
        self.__IS_LABELED = is_labeled
        self.__TARGET_DIR = target_dir
        self.__TARGET_ITEM = []

        if is_labeled is True :
            self.__VAD_DIRPATH = '../../VAD/data/csv/'
            self.__TARGET_ITEM.append('swell')
            self.__TARGET_ITEM.append('serious')
            self.__TARGET_ITEM.append('bite')
            self.__TARGET_ITEM.append('cheerful')
            self.__TARGET_ITEM.append('calm')
            self.__TARGET_ITEM.append('friendly')
            self.__TARGET_ITEM.append('eminent')
        else :
            self.__VAD_DIRPATH = '../../VAD/data/no_label_csv/'
            self.__TARGET_ITEM.append('no_label')

        self.__voice_states = []
        self.__voice_features = []
        self.__fin_times = []
        
        # TODO : read from file
        MAX_ITEM_COUNT = 60
        col_name = ['ID', 'SECTION', 'ANSWER']
        for i in range(1, MAX_ITEM_COUNT + 1) :
            col_name.append(str(i))

    def set_voice_state(self) :
        pathes = glob.glob(self.__VAD_DIRPATH + '*.csv')
        current_target = []
        
        for path in pathes :
            print(path)


def main(is_labeled, target, question_path) :
    path = question_path if is_labeled else ''
    feature = FeatureValue(is_labeled, target, path)
    feature.set_voice_state()

if __name__ == "__main__":
    args = sys.argv
    main(bool(int(args[1])), args[2], '')
