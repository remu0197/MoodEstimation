import film
import os
import datetime
import cv2
import wave
import pygame
from mutagen.mp3 import MP3 as mp3
import time
import pyaudio
import pydub
from pydub import AudioSegment

def check_length() :
    test_path = '../data/test/'

    for i in range(2):
        cap = cv2.VideoCapture(test_path + str(i) + '.avi')            # 動画を読み込む
        video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
        video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する
        video_len_sec = video_frame / video_fps                     # 長さ（秒）を計算する
        print(video_len_sec)                            # 長さ（秒）を出力する

    wf = wave.open(test_path + "0.wav" , "r" )
    print(float(wf.getnframes()) / wf.getframerate())

def play_sound(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    mp3_length = mp3(path).info.length
    pygame.mixer.music.play(1)
    time.sleep(mp3_length + 0.25)
    pygame.mixer.music.stop()

def AdjustVolume(filepath, value=10) :
    base_sound = AudioSegment.from_file(filepath + '/0_r.wav', format="wav")
    adjust_sound = base_sound + value
    adjust_sound.export(filepath + '/result.wav', format='wav')

    base_sound = AudioSegment.from_file(filepath + '/0_l.wav', format="wav")
    adjust_sound = base_sound + value
    adjust_sound.export(filepath + '/result.wav', format='wav')

def main() :
    IS_CHECK_LENGTH = False
    IS_TEST_CAMERA = False
    IS_DEBUG = False
    
    # camera_nums = [1, 2]
    mike_nums = [0, 1, 2, 3]
    Film = film.Film(camera_count=2, mike_count=4, recording_second=300, mike_device_nums=mike_nums)
    Film.setting_sound(channels=2)

    # make dir of experimental data
    path = '../data/'
    if IS_DEBUG == True :
        path = path + 'test/'
    else :
        path = path + '{0:%y%m%d%H%M}'.format(datetime.datetime.now())
        if os.path.exists(path) :
            i = 1

            while os.path.exists(path + '_' + str(i)):
                i = i + 1
            path = path + str(i)

        path = path + '/'
        os.mkdir(path)

    if IS_CHECK_LENGTH == True :
        check_length()
    elif IS_TEST_CAMERA == True :
        Film.capture_test()
    elif Film.start_filming() == True:
        print('now writing sound')
        Film.write_sound(path, devide_stereo=True)
        Film.pause(120)
        print('now encoding videos')
        Film.write_video(path, path + '_sight')
        print('complete')

    return

if __name__ == '__main__': main()