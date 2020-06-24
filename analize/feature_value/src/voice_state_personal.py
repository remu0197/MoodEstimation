import sys
from clusterizer import Clusterizer
import os, datetime

class VoiceStatePersonal :
    def __init__(self, voice_states) :
        self.__voice_states = voice_states
        self.__mean_1, self.__mean_2 = 0, 0

    def __get_personal(self, states) :
        if len(states) is 0 :
            return [], []

        states_personal_1, states_personal_2 = [states[0]], [states[0]]

        for i in range(1, len(states)) :
            state_num = states[i][0]
            state_time = states[i][1]

            # if state_num is 2 :
            #     state_time = state_time / self.__mean_2
            # else :
            #     state_time = state_time / self.__mean_1

            states_personal_1.append([state_num, state_time/self.__mean_1])
            states_personal_2.append([state_num, state_time/self.__mean_2])

        return states_personal_1, states_personal_2


    def __get_means(self, index) :
        states_1, states_2 = [], []
        for states in self.__voice_states :
            for i in range(1, len(states)) :
                state_num = states[i][0]
                state_time = states[i][1]

                if state_num is 1 :
                    states_1.append(state_time)
                elif state_num is 2 :
                    states_2.append(state_time)

        self.__mean_1, self.__mean_2 = self.__get_mean(states_1, index), self.__get_mean(states_2, index)

        # print(str(self.__mean_1) + ' : ' + str(self.__mean_2))

    def __get_mean(self, array, index) :
        if len(array) is 0 : 
            return 1

        cluster = Clusterizer(array)
        labels = cluster.predict(method='SMIRNOV',should_remove_filler=True)
        path = "../fig/" + '{0:%y%m%d%H%M}'.format(datetime.datetime.now())
        if not os.path.exists(path) :
            os.mkdir(path)

        path = path + "/" + str(index) + ".png"
        cluster.save_plot(path)

        # フィラーを外れ値として外す
        # with open(csv_file) as f :
        # reader = csv.reader(f)
        # for i, row in enumerate(reader) :
        #   if row[0] == 1:
        #       labels[i] = 1
        
        total, count = 0, 0
        for i, x in enumerate(array) :
            if labels[i] != 1 :
                total = total + x
                count = count + 1

        if count > 0 :
            return total / count
        
        return 0


    def edit_personal(self, fin_times, ID) :
        voice_states_1, voice_states_2 = [], []
        self.__get_means(ID)
        for states in self.__voice_states :
            states_1, states_2 = self.__get_personal(states)
            voice_states_1.append(states_1)
            voice_states_2.append(states_2)

        fin_times_1, fin_times_2 = [], []
        for i in range(ID*4, (ID+1)*4) :
            fin_times_1.append(fin_times[i] / self.__mean_1)
            fin_times_2.append(fin_times[i] / self.__mean_2)

        return voice_states_1, voice_states_2, fin_times_1, fin_times_2