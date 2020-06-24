# import sys
# import wave
# import pyaudio 
# import concurrent.futures
# import filming

# class RecordingManager:
    
#     def __init__(self, device_count, recording_second):
#         self.__DEVICE_COUNT = device_count
#         self.__RECORDING_SECONd = recording_second

#         self.__FORMAT = pyaudio.paInt16
#         self.__CHANNELS = 1
#         self.__RATE = 44100
#         self.__chunk = 2 ** 11

#         self.__audio = pyaudio.PyAudio()
#         self.__stream_list = []
#         self.__data_list = []

#         for i in range(device_count):
#             stream = self.__audio.open(
#                 format = self.__FORMAT,   
#                 channels = self.__CHANNELS,
#                 rate = self.__RATE,        
#                 input = True,               
#                 input_device_index = i,
#                 frames_per_buffer = self.__chunk 
#             )

#             self.__stream_list.append(stream)

#     def __recording(self, stream):
#         frames = []

#         for _ in range(0, int(self.__RATE / self.__chunk * self.__RECORDING_SECONd)):
#             data = stream.read(self.__chunk)
#             frames.append(data)

#         return frames

#     def __filming(self, stream, func) :
#         frames = []

#         for i, _ in range(0, int(self.__RATE / self.__chunk * self.__RECORDING_SECONd)):
#             if func(i) == False :
#                 return None
#             data = stream.read(self.__chunk)
#             frames.append(data)

#         return frames

#     def start_recording(self):
#         executer = concurrent.futures.ThreadPoolExecutor(max_workers=self.__DEVICE_COUNT)

#         for stream in self.__stream_list:
#             self.__data_list.append(executer.submit(fn=self.__recording, args=stream))

#         return True

#     def start_filming(self, capturing_manager) :
#         if self.__DEVICE_COUNT != capturing_manager.get_device_count() :
#             print('Fail')
#             return False

#         executer = concurrent.futures.ThreadPoolExecutor(max_workers=self.__DEVICE_COUNT * 2)

#         for stream in self.__stream_list :
#             data = executer.submit(self.__filming, stream, capturing_manager.capturing_with_mike)
#             print(data.result())
#             self.__data_list.append(data.result())

#         for data in self.__data_list :
#             if data == None:
#                 return False

#         return True 

#     def write_wavfile(self, base_path):
#         if self.__data_list:
#             for i in range(len(self.__data_list)):
#                 wave_file = wave.open(str(i+1) + '.wav', "w")
#                 wave_file.setnchannels(self.__CHANNELS)
#                 wave_file.setsampwidth(self.__audio.get_sample_size(self.__FORMAT))
#                 wave_file.setframerate(self.__RATE)
#                 wave_file.writeframes(b''.join(self.__data_list[i]))
#                 wave_file.close()

#             return True

#         return False

        

import pyaudio
import concurrent.futures
import wave
import time
import filming_test

class RecordingManager:
    def __init__(self, device_count, recording_second):
        self.__DEVICE_COUNT = device_count
        self.__RECORDING_SECONd = recording_second

        self.__FORMAT = pyaudio.paInt16
        self.__CHANNELS = 2
        self.__RATE = 44100
        self.__chunk = 2 ** 11

    def __recording(self, index, path):
        frames_r = []
        frames_l = []
        
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format = self.__FORMAT,   
            channels = self.__CHANNELS,
            rate = self.__RATE,        
            input = True,               
            input_device_index = 0,
            frames_per_buffer = self.__chunk 
        )

        current_time = time.time()

        # For Debug
        print('device' + str(index) + ':' + str(current_time))

        #正確な時間の録音ができてない
        while time.time() - current_time < self.__RECORDING_SECONd :
            data = stream.read(self.__chunk)
            frames_l.append(data[0::2])
            frames_r.append(data[1::2])

        # For Debug
        print('device' + str(index) + ':' + str(time.time()))

        self.__write_wavfile(frames_l, path, audio, 0)
        self.__write_wavfile(frames_r, path, audio, 1)

        return True

    def __write_wavfile(self, frames, path, audio, index) :
        wave_file = wave.open(path + '/' + str(index) + '.wav', "w")
        wave_file.setnchannels(self.__CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(self.__FORMAT))
        wave_file.setframerate(self.__RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

    def start_recording(self, index, path):
        
        self.__recording(index, path)

        return True
        
