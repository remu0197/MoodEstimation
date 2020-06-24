import csv
import glob
from enum import Enum
import sys
from voice_state import VoiceState
from voice_feature import VoiceFeacure
from distutils.util import strtobool
import os

class FeatureValue :
    def __init__(self, is_labeled) :
        self.__QESTION_FILEPATH = '../questionaire/2018_1211-1220.csv'
        self.__VAD_DIRPATH = '../../VAD/data/no_label_csv/'
        self.__voice_states = []
        self.__voice_features = []
        self.__file_data = open('../data/half/no_label.csv', 'w')
        col_name = ['ID', 'SECTION','ANSWER']

        self.__fin_times = []
        print(self.__VAD_DIRPATH)

        MAX_ITEM_COUNT = 60

        for i in range(1, MAX_ITEM_COUNT + 1) :
            col_name.append(str(i))

        self.__csv_readers = []
        writer = csv.writer(self.__file_data, lineterminator='\n')
        writer.writerow(col_name)
        self.__csv_readers.append(writer)

    def set_voice_states(self) :
        dirs = os.listdir(self.__VAD_DIRPATH)
        for dir in dirs :
            self.set_voice_state(dir=dir+'/')

    def set_voice_state(self, dir='') :
        pathes = glob.glob(self.__VAD_DIRPATH + dir + '*.csv')
        current_targets = []

        for i, path in enumerate(pathes) :
            current_targets.append(path)

            if len(current_targets) == 2 :
                print("current: " + str(current_targets))
                path_l = current_targets[0]
                path_r = current_targets[1]

                # get left person's data
                # file_id = int(current_targets[0].strip(self.__VAD_DIRPATH + '\\').strip(".csv"))
                # section = int(file_id / 1000)
                # subject_id = int(file_id % 1000)
    
                voice_l = VoiceState(path_l, path_r)
                self.__voice_states.append(voice_l.get_voice_states(i-1, i-1))
                self.__fin_times.append(voice_l.get_fin_time())

                voice_r = VoiceState(path_r, path_l)
                self.__voice_states.append(voice_r.get_voice_states(i, i))
                self.__fin_times.append(voice_r.get_fin_time())

                current_targets.clear()  

        return True

    def set_voice_features(self) :
        for states in self.__voice_states :
            for i, state in enumerate(states) :
                if len(state) > 0 :
                    print("state: " + str(state))
                    feature = VoiceFeacure(voice_states=state)
                    self.__voice_features.append(feature.get_voice_feature(self.__fin_times[i]))

    def append_answer_features(self) :
        f = open(self.__QESTION_FILEPATH, 'r')
        reader = csv.reader(f)
        for row in reader :
            subject_num = int(row[0])

            for i in range(4) :
                feature = []
                for vf in self.__voice_features :
                    if vf[0] is subject_num and vf[1] is (i + 1):
                        feature = vf

                if len(feature) == 0 :
                    print(str(subject_num) + '-' + str(i+1) + '\'s data is lost')
                    continue

                for j in range(1, 8) :
                    target = [subject_num, (i + 1), row[i * 7 + j]]
                    target.extend(feature[2:])
                    self.__csv_readers[j-1].writerow(target)     

        f.close()             

    def close(self) :
        for f in self.__file_data :
            f.close()

    def write(self, path) :
        f = open(path, 'w')
        writer = csv.writer(f, lineterminator='\n')
        for features in self.__voice_features :
            writer.writerow(features)

        f.close()
        return True

def main(is_labeled=True) :
    feature = FeatureValue(is_labeled)
    # if is_labeled == True :
    #     feature.set_voice_state()
    # else :
    feature.set_voice_states()

    feature.set_voice_features()
    # if is_labeled == True :
    #     feature.append_answer_features()
    feature.close()

if __name__ == '__main__':
    args = sys.argv
    main()