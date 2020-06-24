import sys

class VoiceFeacure :
    def __init__(self, path='', voice_states=None, is_half=False) :
        self.__states = [[] for i in range(4)]
        self.__results = []
        self.__IS_HALF = is_half
        # 変更
        self.__STATISTIC_COUNT = 5

        if voice_states is not None :
            self.__subject_num = voice_states[0][0]
            self.__section_num = voice_states[0][1]

            for i in range(1, len(voice_states)) :
                num = voice_states[i][0]
                time = voice_states[i][1]
                self.__states[num - 1].append(time)

            if is_half :
                self.__states.pop(1)

        elif path is not '' :
            f = open(path, 'r')
            f.close()
        else :
            print('Can not set VoiceStatistics')
            sys.exit()

    def get_voice_feature(self, fin_time) :
        results = []
        # No.001 - 024
        self.__get_voice_statistics(self.__states, results, fin_time)
        # No.025 - 114
        if self.__IS_HALF :
            self.__add_half_voice_feature(len(self.__states), results)
        else :
            self.__add_voice_features(len(self.__states), results)

        self.__results.extend(results)

        return self.__results

    def __get_voice_statistics(self, state, results, fin_time) :
        results.extend(self.__mean(state))
        # results.extend(self.__var(state, results))
        results.extend(self.__min(state))
        results.extend(self.__max(state))
        results.extend(self.__count(state, fin_time))
        results.extend(self.__occupy(state))

    def __mean(self, state) :
        results = []

        for data in state:
            result = 0.0
            if len(data) > 0 :
                for time in data:
                    result += time
                result = result / len(data)
            results.append(result)

        return results

    def __var(self, state, mean) :
        results = []
        i = 0
        for data in state:
            result = 0.0
            if len(data) > 0 :
                for time in data:
                    x = (time - mean[i])
                    result += x * x
                result = result / len(data)
            results.append(result)
            i = i + 1

        return results

    def __min(self, state) :
        results = []

        for data in state:
            min = 1000.0
            for time in data:
                if time < min:
                    min = time
            if min is 1000.0 :
                min = 0
                
            results.append(min)

        return results

    def __max(self, state) :
        results = []

        for data in state:
            max = 0.0
            for time in data:
                if time > max:
                    max = time
            results.append(max)

        return results

    def __count(self, state, fin_time) :
        results = []

        for data in state:
            result = len(data) / fin_time
            results.append(result)

        return results

    def __occupy(self, state) :
        results = [0.0 for _ in range(len(state))]
        total_time = 0.0
        i = 0

        for data in state:
            for time in data:
                results[i] += time
                total_time += time
            i = i + 1

        for i in range(len(results)):
            results[i] = results[i] / total_time

        return results

    def __add_voice_features(self, state_count, results) :
        if not self.__IS_HALF :
            # No.25-30
            # for i in range(self.__STATISTIC_COUNT):
            #     s1_index = i * state_count
            #     x = results[s1_index + 1] / results[s1_index]
            #     results.append(x)

            # No.31-36
            for i in range(self.__STATISTIC_COUNT):
                s1_index = i * state_count
                x = (results[s1_index + 1] + results[s1_index + 2]) / (results[s1_index] + results[s1_index + 2])
                results.append(x)

            # No.37-42
            for i in range(self.__STATISTIC_COUNT):
                s1_index = i * state_count
                x = (results[s1_index]) / (results[s1_index] + results[s1_index + 1])
                results.append(x)

            # No.43-48
            for i in range(self.__STATISTIC_COUNT):
                s1_index = i * state_count
                x = (results[s1_index + 1]) / (results[s1_index] + results[s1_index + 1])
                results.append(x)

        # No.49-54
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 3] / (results[s1_index] + results[s1_index + 1] + results[s1_index + 2])
            results.append(x)

        totals = []
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index] + results[s1_index + 1] + results[s1_index + 2] + results[s1_index + 3]
            totals.append(x)

        # No.55-60
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index] / totals[i]
            results.append(x)

        # No.61-66
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 1] / totals[i]
            results.append(x)

        # No.67-72
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 2] / totals[i]
            results.append(x)

        # No.73-78
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 3] / totals[i]
            results.append(x)

        # No.79-84
        # for i in range(self.__STATISTIC_COUNT):
        #     s1_index = i * state_count
        #     x = results[s1_index + 2] / results[s1_index]
        #     results.append(x)

        # No.85-90
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index + 2] / (results[s1_index] + results[s1_index + 2])
            results.append(x)

        # No.91-96
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index] / (results[s1_index] + results[s1_index + 2])
            results.append(x)

        # No.97-102
        # for i in range(self.__STATISTIC_COUNT):
        #     s1_index = i * state_count
        #     print(str(i) + ': ' + str(results[s1_index + 1]))
        #     x = results[s1_index + 2] / results[s1_index + 1]
        #     results.append(x)

        # No.103-108
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index + 1] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index + 2] / (results[s1_index + 1] + results[s1_index + 2])
            results.append(x)

        # No.109-114
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index + 1] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index + 1] / (results[s1_index + 1] + results[s1_index + 2])
            results.append(x)

    def __add_half_voice_feature(self, state_count, results) :
        # No.49-54
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 3] / (results[s1_index] + results[s1_index + 2])
            results.append(x)

        totals = []
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index] + results[s1_index + 2] + results[s1_index + 3]
            totals.append(x)

        # No.55-60
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index] / totals[i]
            results.append(x)

        # No.61-66
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 2] / totals[i]
            results.append(x)

        # No.67-72
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 3] / totals[i]
            results.append(x)

        # No.79-84
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = results[s1_index + 2] / results[s1_index]
            results.append(x)

        # No.85-90
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index + 2] / (results[s1_index] + results[s1_index + 2])
            results.append(x)

        # No.91-96
        for i in range(self.__STATISTIC_COUNT):
            s1_index = i * state_count
            x = 0
            if results[s1_index] != 0 or results[s1_index + 2] != 0 :
                x = results[s1_index] / (results[s1_index] + results[s1_index + 2])
            results.append(x)