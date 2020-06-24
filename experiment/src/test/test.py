# import sys
# import time

# str = ''

# for i in range(100):
#     if i % 3 == 0 :
#         str = ''
#     else :
#         str += '.'

#     sys.stdout.write("\r%s" % str)
#     sys.stdout.flush()
#     time.sleep(0.5)

# import cv2

# cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1600,900))

# while True:
#     ret, frame = cap.read()

#     frame = frame[::-1,:]

#     if ret == True:
#         frame = cv2.flip(frame, 0)

#         out.write(frame)

#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     else :
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# from matplotlib import pyplot

# pyplot.plot(1, 1)
# pyplot.show()

# import pyaudio
# import time
# import wave

# __DEVICE_COUNT = 1
# __RECORDING_SECONd = 10

# __FORMAT = pyaudio.paInt16
# __CHANNELS = 1
# __RATE = 44100
# __chunk = 2 ** 11

# __audio = pyaudio.PyAudio()
# __stream_list = []
# __data_list = []

# stream = __audio.open(
#     format = __FORMAT,   
#     channels = __CHANNELS,
#     rate = __RATE,        
#     input = True,               
#     input_device_index = 1,
#     frames_per_buffer = __chunk 
#     )

# # current_time = time.time()
# # frames_r = []
# # frames_l = []

# # while time.time() - current_time < 10 :
# #     frame = stream.read(__chunk)
# #     frame_r = frame[0::2]
# #     frame_l = frame[1::2]
# #     frames_r.append(frame_r)
# #     frames_l.append(frame_l)


# # wave_file = wave.open(str(0) + '.wav', "w")
# # wave_file.setnchannels(1)
# # wave_file.setsampwidth(__audio.get_sample_size(__FORMAT))
# # wave_file.setframerate(__RATE)
# # wave_file.writeframes(b''.join(frames_l))
# # wave_file.close()

# # wave_file = wave.open(str(1) + '.wav', "w")
# # wave_file.setnchannels(1)
# # wave_file.setsampwidth(__audio.get_sample_size(__FORMAT))
# # wave_file.setframerate(__RATE)
# # wave_file.writeframes(b''.join(frames_r))
# # wave_file.close()

# import os
# import datetime
# import recording_test
# import filming_test
# import concurrent.futures
# import wave
# import cv2
# import pyaudio
# import time

# def main():
#     IS_DEBUG = True
#     DEVICE_COUNT = 1            # mike x 2, camera x 2
#     RECOURDING_SECOND = 10     # 3 min

#     # make dir of current data
#     dir_name = '{0:%y%m%d%H%M}'.format(datetime.datetime.now())
#     path = '../data/' + 'test' # dir_name
#     # os.mkdir(path)

#     # # 先にmainでマイクとかopenしてから並列化

#     # mike = recording_test.RecordingManager(DEVICE_COUNT, RECOURDING_SECOND)
#     camera = filming_test.CapturingManager(path, RECOURDING_SECOND)

#     executer = concurrent.futures.ThreadPoolExecutor(2)
#     # executer.submit(mike.start_recording, 0, path)
    
#     executer.submit(camera.capture, path, 0)
#     # executer.submit(camera.capture, path, 1)

#     # executer.submit(camera.capture, path)

#     #filming_test.func(RECOURDING_SECOND, path)

#     check_length()


#     return 

# def check_length() :
#     test_path = '../data/test/0'

#     cap = cv2.VideoCapture(test_path + '.avi')            # 動画を読み込む
#     video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
#     video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する
#     video_len_sec = video_frame / video_fps                     # 長さ（秒）を計算する
#     print(video_len_sec)                            # 長さ（秒）を出力する

#     # wf = wave.open(test_path + ".wav" , "r" )
#     # print(float(wf.getnframes()) / wf.getframerate())

# def print_time() :
#     current_time = time.time()

#     while True:
#         print(time.time() - current_time)
#         if cv2.waitKey(25) & 0xFF == ord('q') :
#             break

# def open_mike(count) :
#     mikes = []
#     audio = pyaudio.PyAudio()

#     for i in range(count) :
#         stream = audio.open(
#             format = pyaudio.paInt16,
#             cannels = 1,
#             rate = 44100,
#             input = True,
#             input_device_index = i,
#             frames_per_buffer = 2 ** 11
#         )

#         mikes.append(stream)

#     return mikes


# if __name__ == '__main__': main()

import pyaudio
import wave

audio = pyaudio.PyAudio()

for i in range(0, audio.get_device_count()) :
    print(audio.get_device_info_by_index(i))

# import audioop

# test = {2000: 1}

# test[200] = 0

# for k in test :
#     print(k)

# import numpy as np
# import scipy.io.wavfile as wav 

# def main():
#     # データのパラメータ
#     N = 256            # サンプル数
#     dt = 0.01          # サンプリング間隔
#     f1, f2 = 10, 20    # 周波数
#     t = np.arange(0, N*dt, dt) # 時間軸
#     freq = np.linspace(0, 1.0/dt, N) # 周波数軸

#     wav.read()

#     # 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
#     f = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 0.3 * np.random.randn(N)

#     # 高速フーリエ変換
#     F = np.fft.fft(f)

#     # 振幅スペクトルを計算
#     Amp = np.abs(F)
