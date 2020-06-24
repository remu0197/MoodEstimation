import scipy.io.wavfile as wav 
import math
import numpy as np

class VAD :
    def __init__(self) :
        self.__source = None
        self.__fs = None
        self.__context = None

        # vad parameters
        self.__fft_size = 512
        self.__buffer_len = 512
        self.__smoothing_time_constant = 0.99
        self.__energy_offset = 1e-8
        self.__energy_threshold_ratio_pos = 2
        self.__energy_threshold_ratio_neg = 0.5
        self.__energy_threshold_pos = self.__energy_offset * self.__energy_threshold_ratio_pos
        self.__energy_threshold_neg = self.__energy_offset * self.__energy_threshold_ratio_neg
        self.__energy_integration = 1
        self.__filter = []
        self.__is_ready_energy = False
        self.__is_vad_state = False
        self.__energy = 0
        
        self.__voice_trend = 0
        self.__voice_trend_max = 10
        self.__voice_trend_min = -10
        self.__voice_trend_start = 5
        self.__voice_trend_end = -5

        # for __set_filter()
        self.__option_filter = {200: 0, 2000: 1}

        self.__hertz_per_bin = 0
        self.__iteration_frequency = 0
        self.__iteration_period = 0

        self.__frequency_bin_count = math.floor(self.__fft_size / 2)
        self.__float_frecuency_data = np.zeros(self.__frequency_bin_count)
        self.__float_frequency_data_linear = np.zeros(self.__frequency_bin_count)


    def vad(self, path, is_debug=False) :
        if path == '':
            print('please input data path')
            return False

        self.__read_source(path)

        if is_debug == True :
            self.__debug_message()

        self.__set_filter()
        
        return True

    def __update(self) :
        fft = self.__float_frecuency_data
        for i in range(len(fft)) :
            self.__float_frequency_data_linear[i] = math.pow(10, fft[i] / 10)
        self.__is_ready_energy = False

    def __getEnegy(self) :
        if(self.__is_ready_energy) :
            return

        self.__energy = 0
        fft = self.__float_frequency_data_linear

        for i in range(len(fft)) :
            self.__energy += self.__filter[i] * fft[i] * fft[i]

        self.__is_ready_energy = True

        return self.__energy

    def monitor(self) :
        signal = self.__energy - self.__energy_offset

        if signal > self.__energy_threshold_pos :
            if self.__voice_trend + 1 > self.__voice_trend_max :
                self.__voice_trend = self.__voice_trend_max
            else :
                self.__voice_trend = self.__voice_trend + 1
        elif signal < self.__energy_threshold_neg :
            if self.__voice_trend - 1 < self.__voice_trend_min :
                self.__voice_trend_min
            else :
                self.__voice_trend - 1
        else :
            if self.__voice_trend > 0 :
                self.__voice_trend = self.__voice_trend - 1
            elif self.__voice_trend < 0 :
                self.__voice_trend = self.__voice_trend + 1

        is_voice_start = False
        is_voice_end = False

        if self.__voice_trend > self.__voice_trend_start :
            is_voice_start = True
        elif self.__voice_trend < self.__voice_trend_end :
            is_voice_end = True

        integrasion = signal * self.__iteration_period * self.__energy_integration

        if integrasion > 0 or is_voice_end == False :
            self.__energy_offset = self.__energy_offset + integrasion
        else :
            self.__energy_offset = self.__energy_offset + integrasion * 10
        
        if self.__energy_offset < 0 :
            self.__energy_offset = 0
        self.__energy_threshold_pos = self.__energy_offset * self.__energy_threshold_ratio_pos
        self.__energy_threshold_neg = self.__energy_offset * self.__energy_threshold_ratio_neg

        if is_voice_start == True and self.__is_vad_state = False :
            self.__is_vad_state = True
            self.__voice_start()
        if is_voice_end == True and self.__is_vad_state = True :
            self.__is_vad_state = False 
            self.__voice_stop()

        return signal
    
    def __voice_start(self) :
        return

    def __voice_end(self) :
        return

    def __read_source(self, path) :
        fs, s = wav.read(path)
        fs = 48000
        self.__fs = fs
        self.__source = s
        print(len(s[0]))

        self.__hertz_per_bin = fs / self.__fft_size
        self.__iteration_frequency = fs / self.__buffer_len
        self.__iteration_period = 1 / self.__iteration_frequency

    def __set_filter(self) :
        self.__filter.clear()
        for i in range(math.floor(self.__fft_size / 2)) :
            self.__filter.append(0)
            for j in self.__option_filter :
                if i * self.__hertz_per_bin < j :
                    self.__filter[i] = self.__option_filter[j]
                    break
        
        print(self.__filter)
        return

    def __debug_message(self) :
        print('Vad')
        print('| sampleRate         : ' + str(self.__fs))
        print('| hertzPerBin        : ' + str(self.__hertz_per_bin))
        print('| iterationFrequency : ' + str(self.__iteration_frequency))
        print('| iterationPeriod    : ' + str(self.__iteration_period))

if __name__ == '__main__': 
    # example of execute VAD
    vad = VAD()
    vad.vad(is_debug=True, path='../../experiment/data/1810161635/3.wav')