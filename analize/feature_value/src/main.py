import csv
import os
import glob
from enum import Enum
import sys
from voice_state import VoiceState
from voice_feature import VoiceFeacure
from voice_state_personal import VoiceStatePersonal
import pandas as pd

class FeatureValue :
    def __init__(self, dirname, is_both=True) :
        self.__QESTION_FILEPATH = '../../processing/data/questionaire/all.csv'
        self.__VAD_DIRPATH = '../../processing/data/VAD/data/csv/'
        self.__IS_BOTH = is_both
        self.__voice_states = []
        self.__voice_features = []
        self.__personal_states_1 = []
        self.__personal_states_2 = []

        basepath = '../data/' + dirname + '/'
        if not os.path.exists(basepath) :
            os.mkdir(basepath)

        self.__file_data = [
            # open('../data/half/no_label.csv', 'w'),
            open(basepath + 'swell.csv', 'w'),
            open(basepath + 'serious.csv', 'w'),
            open(basepath + 'bite.csv', 'w'),
            open(basepath + 'cheerful.csv', 'w'),
            open(basepath + 'calm.csv', 'w'),
            open(basepath + 'friendly.csv', 'w'),
            open(basepath + 'eminent.csv', 'w')
        ]

        self.__fin_times = [[], [], []]

        MAX_ITEM_COUNT = 240
        col_name = ['ANSWER', 'ID', 'SECTION']
        for i in range(1, MAX_ITEM_COUNT + 1) :
            col_name.append(str(i))

        # countなしで0からインデクシングしてくれる方法
        # features = pd.DataFrame()
        # one_feature = pd.Series(data=feature)
        # one_feature = one_feature.append(subject_info)
        # features = features.append(one_feature, ignore_index=True)

        self.__csv_readers = []
        for f in self.__file_data :
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(col_name)
            self.__csv_readers.append(writer)

    def set_voice_state(self) :
        pathes = glob.glob(self.__VAD_DIRPATH + '*.csv')
        current_targets = []

        for path in pathes :
            current_targets.append(path)

            if len(current_targets) == 2 :
                # print(current_targets)
                path_l = current_targets[0]
                path_r = current_targets[1]

                # get left person's data
                file_id = int(current_targets[0].lstrip(self.__VAD_DIRPATH + '\\').rstrip(".csv"))
                section = int(file_id / 1000)
                subject_id = int(file_id % 1000)
                # print(subject_id)
                while len(self.__voice_states) < subject_id + 1 :
                    self.__voice_states.append([[],[],[],[]])
                voice_l = VoiceState(path_l, path_r)
                self.__voice_states[subject_id][section - 1] = voice_l.get_voice_states(subject_id, section, self.__IS_BOTH)
                self.__fin_times[0].append(voice_l.get_fin_time())

                # get right person's data
                file_id_base = current_targets[1].lstrip(self.__VAD_DIRPATH + '\\').rstrip(".csv")
                if not "_p" in file_id_base :
                    file_id = int(file_id_base)
                    section = int(file_id / 1000)
                    subject_id = int(file_id % 1000)
                    # print(subject_id)
                    while len(self.__voice_states) < subject_id + 1 :
                        self.__voice_states.append([[],[],[],[]])
                    voice_r = VoiceState(path_r, path_l)
                    self.__voice_states[subject_id][section - 1] = voice_r.get_voice_states(subject_id, section, self.__IS_BOTH)
                    self.__fin_times[0].append(voice_r.get_fin_time())

                current_targets.clear()

        return True

    def write_voice_state(self) :
        for states in self.__voice_states:
            for state in states:
                if len(state) == 0: 
                    continue

                subject = state[0]
                filepath = '../../processing/data/state_list/' + str(subject[0]) + '_' + str(subject[1]) + '.csv'

                with open(filepath, 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(state[1:])

    def add_voice_state_personal(self) :
        # print(len(self.__voice_states))
        subject_count = 0
        for i, states in enumerate(self.__voice_states) :
            if len(states[0]) is 0 :
                print('No.' + str(i) + ' is LOST')
                continue

            personal = VoiceStatePersonal(states)
            # self.__personal_states.append(personal.get_personal(self.__fin_times[0][i]))
            states_1, states_2, fin_1, fin_2 = personal.edit_personal(self.__fin_times[0], subject_count)
            self.__personal_states_1.append(states_1)
            self.__personal_states_2.append(states_2)
            self.__fin_times[1].extend(fin_1)
            self.__fin_times[2].extend(fin_2)
            subject_count += 1

        return True

    # TODO fin_timeどうするか + extend ２回ののちappend
    def set_voice_features(self) :
        index = 0
        for x,states in enumerate(self.__voice_states) :
            is_exist = False
            for state in states :
                if len(state) > 0 :
                    is_exist = True

            if is_exist is False :
                continue
            
            for i, state in enumerate(states) :
                features = [x, i+1]
                if len(state) > 0 :
                    # print(len(self.__personal_states_1[x]))
                    feature = VoiceFeacure(voice_states=state, is_half=not self.__IS_BOTH)
                    features.extend(feature.get_voice_feature(self.__fin_times[0][index]))

                    feature = VoiceFeacure(voice_states=self.__personal_states_1[int(index/4)][i])
                    features.extend(feature.get_voice_feature(self.__fin_times[1][index]))

                    feature = VoiceFeacure(voice_states=self.__personal_states_2[int(index/4)][i])
                    features.extend(feature.get_voice_feature(self.__fin_times[2][index]))
                    
                    self.__voice_features.append(features)

                    index = index + 1
                    
    def append_answer_features(self) :
        f = open(self.__QESTION_FILEPATH, 'r')
        reader = csv.reader(f)
        for row in reader :
            for answer in row :
                if answer is '' :
                    continue

            subject_num = int(row[0])

            for i in range(4) :
                feature = []
                for vf in self.__voice_features :
                    if vf[0] is subject_num and vf[1] is (i + 1):
                        feature = vf
                        break

                if len(feature) == 0 :
                    print(str(subject_num) + '-' + str(i+1) + '\'s data is lost')
                    continue

                for j in range(1, 8) :
                    target = [row[i * 7 + j]]
                    target.extend(feature)
                    self.__csv_readers[j-1].writerow(target)     

        f.close()             

    def close(self) :
        for f in self.__file_data :
            f.close()

    def write(self) :
        # f = open(path, 'w')
        # writer = csv.writer(f, lineterminator='\n')
        # for features in self.__voice_features :
        #     writer.writerow(features)

        # f.close()
        for csv in self.__csv_readers :
            for features in self.__voice_features :
                csv.writerow(features)
        return True

def main() :
    feature = FeatureValue(dirname='JSKE_2020_3', is_both=True)
    feature.set_voice_state()
    feature.write_voice_state()
    feature.add_voice_state_personal()
    feature.set_voice_features()
    feature.append_answer_features()
    # feature.write()
    feature.close()

if __name__ == '__main__':
    main()