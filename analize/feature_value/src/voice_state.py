import csv

class Person() :
    FIRST = 0
    SECOND = 1
    THIRD = 2

class State() :
    START = 0
    STOP = 1

class VoiceState :
    def __init__(self, path_f, path_s) :
        self.__stops = []
        self.__is_stop = [True, True]
        self.__is_fin_search = [False, False]
        self.__index = [0, 0]
        self.__voice_states = []
        self.__time_line = 0.0
        self.__stash_timeline = -1

        stops_f = []
        stops_s = []

        # Load first person's data
        f = open(path_f, 'r')
        # print(path_f)
        reader = csv.reader(f)
        for row in reader :
            stops_f.append([float(s) for s in row])
        f.close()

        # Load second person's data
        f = open(path_s, 'r')
        # print(path_s)
        reader = csv.reader(f)
        for row in reader :
            stops_s.append([float(s) for s in row])
        f.close()

        self.__fin_time = stops_f[len(stops_f)-1][1] if stops_f[len(stops_f)-1][1] > stops_s[len(stops_s)-1][1] else stops_s[len(stops_s)-1][1]

        self.__stops.append(stops_f)
        self.__stops.append(stops_s)
        f.close()

    def get_voice_states(self, subject_id, section_num, is_both=True) :
        self.__voice_states.append([subject_id, section_num])
        top_first = self.__stops[Person.FIRST][0][State.START]
        top_second = self.__stops[Person.SECOND][0][State.START]
        self.__time_line = top_first if top_first > top_second else top_second
        
        while self.__index[Person.FIRST] < len(self.__stops[Person.FIRST]) or self.__index[Person.SECOND] < len(self.__stops[Person.SECOND]) :
            if self.__is_stop[Person.FIRST] and self.__is_stop[Person.SECOND] :
                # Nobady is talk(State No.4)
                self.__stack_voice_state(4)
            elif not(self.__is_stop[Person.FIRST]) and self.__is_stop[Person.SECOND] :
                # First is talk(State No.1)
                self.__stack_voice_state(1)
            elif self.__is_stop[Person.FIRST] and not(self.__is_stop[Person.SECOND]) :
                # Second is talk(State No.2)
                self.__stack_voice_state(2, is_both)
            else :
                # Both are talk(State No.3)
                self.__stack_voice_state(3)

        if not is_both :
            self.__voice_states = self.__combine_silent_state(self.__voice_states)

        return self.__voice_states
    
    def get_fin_time(self) :
        return self.__fin_time

    def __stack_voice_state(self, state_num, is_valid=True) :
        state_f = State.STOP if self.__is_stop[Person.FIRST] else State.START
        state_s = State.STOP if self.__is_stop[Person.SECOND] else State.START
        index_f = self.__index[Person.FIRST]
        index_s = self.__index[Person.SECOND]
    
        next_f = self.__get_next_voice(Person.FIRST, state_f)
        next_s = self.__get_next_voice(Person.SECOND, state_s)

        if next_f < next_s :
            self.__update_voice_state(state_num, Person.FIRST, index_f, state_f, next_f, is_valid)
        elif next_f > next_s :
            self.__update_voice_state(state_num, Person.SECOND, index_s, state_s, next_s, is_valid)
        else :
            if not(self.__is_fin_search[Person.FIRST]) :
                self.__update_voice_state(state_num, Person.FIRST, index_f, state_f, next_f, is_valid)
            if not(self.__is_fin_search[Person.SECOND]) :
                self.__update_voice_state(state_num, Person.SECOND, index_s, state_s, next_s, is_valid)


    def __update_voice_state(self, state_num, person, index, state, next, is_valid=True) :
        # If don't need First voice state, don't convert state.
        if not is_valid :
            self.__stash_timeline = next
        else :
            next_timeline = next
            if self.__stash_timeline >= 0.0 :
                if state_num is 4 :
                    next_timeline = self.__stash_timeline
                else :
                    self.__export_voice_state(self.__stash_timeline, 4)
                self.__stash_timeline = -1.0

            self.__export_voice_state(next_timeline, state_num)

        self.__is_stop[person] = not(self.__is_stop[person])
        if state is State.STOP :
            self.__index[person] = index + 1

    def __export_voice_state(self, next, state_num) :
        state_length = next - self.__time_line
        if state_length > 0.0 :
            self.__voice_states.append([state_num, state_length])

        self.__time_line = next

    def __get_next_voice(self, person, state) :
        index = self.__index[person]
        if index >= len(self.__stops[person]) :
            self.__is_fin_search[person] = True
            return self.__fin_time
        else :
            # print(str(person) + ' ' + str(index) + ' ' + str(state))
            # print(str(len(self.__stops)) + ' ' + str(len(self.__stops[person])) + ' ' + str(len(self.__stops[person][index])))
            return self.__stops[person][index][state]

    def __combine_silent_state(self, states) :
        is_silent = False
        result = [states[0]]
        temp = 0.0

        for i in range(1, len(states)) :
            if states[i][0] is 4 :
                temp += states[i][1]
                is_silent = True
                continue
            elif is_silent is True :
                result.append([4, temp])
                is_silent = False
                temp = 0.0

            result.append(states[i])

        return result