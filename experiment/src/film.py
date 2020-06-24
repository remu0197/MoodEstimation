import cv2
import time
import concurrent.futures
import pyaudio
import wave
import sys
import audioop
import pygame
from mutagen.mp3 import MP3 as mp3

class Film :
    def __init__(self, camera_count=1, mike_count=1, recording_second=10, camera_device_nums=[], mike_device_nums=[], fps=30.0) :
        self.__RECORDING_SECOND = recording_second
        self.__CAMERA_FRAMES = recording_second * fps
        self.__CAMERA_COUNT = camera_count
        self.__MIKE_COUNT = mike_count
        self.__is_write_log = False
        # print('Recording second: ' + str(self.__RECORDING_SECOND))

        if recording_second != 300 :
            print('This is test of experiment ?')
            answer = input("Answer: ")
            if answer != "yes":
                print('Please change recording_second')
                sys.exit()

        self.__caps = []
        if len(camera_device_nums) == camera_count:
            for i in camera_device_nums :
                cap = cv2.VideoCapture(i)
                self.__caps.append(cap)
        else :
            for i in range(camera_count):
                cap = cv2.VideoCapture(i)
                self.__caps.append(cap)
        
        self.__video_frames = []
        self.__base_sight_frames = []

        self.__format = pyaudio.paInt16
        self.__channels = 1
        self.__rate = 48000
        self.__chunk = 2 ** 11
        self.__audio = pyaudio.PyAudio()
        self.__sound_frames = []

        self.__logs = ''

        self.__streams = []
        if mike_count == len(mike_device_nums) :
            for i in mike_device_nums:
                print(i)
                stream = self.__audio.open(
                    format=self.__format,
                    channels=self.__channels,
                    rate=self.__rate,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=self.__chunk
                )

                self.__streams.append(stream)
        else :
            for i in range(mike_count):
                stream = self.__audio.open(
                    format=self.__format,
                    channels=self.__channels,
                    rate=self.__rate,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=self.__chunk
                )

                self.__streams.append(stream)

    def __boot_test(self, boot_count) :
        for _ in range(boot_count) :
            # camera test
            for cap in self.__caps :
                ret = self.__capture_boot_test(cap, 1)
                if ret == False :
                    return False
            # mike test
            for stream in self.__streams :
                ret = self.__record_boot_test(stream, 1)
                if ret == False :
                    return False

        return True

    def __capture_base_sight(self, capture_length=15.0) :
        i = 1
        for cap in self.__caps :
            print('Please input enter. Start capture base sight' + str(i))
            input()
            start_time = time.time()
            frames = []
            while time.time() - start_time < capture_length :
                _, frame = cap.read()
                frames.append(frame)

            self.__base_sight_frames.append(frames)
            i = i + 1

        print('base sight has captured. please qlose camera windows.')

        return True

    def __check_camera(self) :
        executer = concurrent.futures.ThreadPoolExecutor(self.__CAMERA_COUNT + 1)
        i = 0
        results = [None, None, None]
        for cap in self.__caps :
            results[i] = executer.submit(self.func, cap, i)
            i = i + 1

        results[2] = executer.submit(self.__capture_base_sight)

        for result in results :
            result.result()

    # use when you want to see from device in real time
    def capture_test(self) :
        executer = concurrent.futures.ThreadPoolExecutor(self.__CAMERA_COUNT + self.__MIKE_COUNT)
        i = 0

        for cap in self.__caps :
            executer.submit(self.func, cap, i)
            i = i + 1

    def func(self, cap, index) :
        print('test')
        while True :
            _, frame = cap.read()
            cv2.imshow('frame' + str(index), frame)

            if cv2.waitKey(1) & 0xFF== ord('q') :
                break

        return True

    def __operation_check(self, start_time) :
        out = ''
        dots = ''
        i = 0
        interval = self.__RECORDING_SECOND / 10
        next_time = interval

        while time.time() - start_time < self.__RECORDING_SECOND :
            time.sleep(0.3)
            if i < 3 :
                dots += '.'
                i = i + 1
            else :
                dots = ''
                i = 0

            if time.time() - start_time > next_time :
                out += '|'
                next_time += interval

            sys.stdout.write("\r%s" % out + dots)
            sys.stdout.flush()

        sys.stdout.write("\r%s" % out)
        sys.stdout.flush()
        print(' ')

    def pause(self, second) :
        start_time = int(time.time())
        elapsed_time = 0
        while elapsed_time < second :
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write("\r%s" % 'pause : ' + str(elapsed_time) + ' / ' + str(second))
            sys.stdout.flush()
            continue

    # For Boot Test
    def __capture_boot_test(self, cap, test_second) :
        start_time = time.time()
        while time.time() - start_time < test_second :
            ret, frame = cap.read()
            if ret == False:
                return False

    def __capture(self, cap, start_time, index) :
        frames = []
        count = 0
        self.__logs += 'capture start at: ' + str(time.time()) + '\n'

        while count < self.__CAMERA_FRAMES :
            _, frame = cap.read()
            frames.append(frame)
            count = count + 1

        self.__logs += 'capture finish at: ' + str(time.time()) + '\n'

        return frames

    # For Boot Test
    def __record_boot_test(self, stream, test_second) :
        start_time = time.time()
        while time.time() - start_time < test_second :
            frame = stream.read(self.__chunk)
            if frame == None :
                return False

        return True

    def __record(self, stream, start_time) :
        frames = []
        self.__logs += 'record start at: ' + str(time.time()) + '\n'

        while time.time() - start_time < self.__RECORDING_SECOND :
            frame = stream.read(self.__chunk)
            frames.append(frame)

        self.__logs += 'record finish at: ' + str(time.time()) + '\n'

        return frames

    # For CLI
    def __ready_to_start(self) :
        for i in range(3):
            print('Ready ...(please input \'start\' or \'quit\')')
            print(':', end=' ')

            str = input()
            if str == 'start' :
                return True
            elif str == 'quit' :
                return False

        return False
        
    def start_filming(self) :
        if self.__boot_test(1) == False :
            print('Fail to Film')
            return False

        self.__check_camera()

        if self.__ready_to_start() == False :
            return False

        total_device_count = self.__MIKE_COUNT + self.__CAMERA_COUNT

        executer = concurrent.futures.ThreadPoolExecutor(total_device_count + 1)
    
        video_results = []
        for i in range(self.__CAMERA_COUNT) :
            video_results.append(None)
        sound_results = []
        for i in range(self.__MIKE_COUNT) :
            sound_results.append(None)

        self.__play_sound('./sound/fin.mp3')

        start_time = time.time()

        pause = executer.submit(self.__operation_check, start_time)

        i = 0
        for cap in self.__caps :
            video_results[i] = executer.submit(self.__capture, cap, start_time, i)
            i = i + 1

        i = 0
        for stream in self.__streams :
            sound_results[i] = executer.submit(self.__record, stream, start_time)
            i = i + 1

        for result in video_results :
            self.__video_frames.append(result.result())
        for result in sound_results :
            self.__sound_frames.append(result.result())

        pause.result()

        self.__play_sound('./sound/fin.mp3')

        return True

    def __play_sound(self, path) :
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        mp3_length = mp3(path).info.length
        pygame.mixer.music.play(1)
        time.sleep(mp3_length + 0.25)
        pygame.mixer.music.stop()

    # channels/ 1:Monaural, 2:Stereo
    def setting_sound(self, format=pyaudio.paInt16,channels=1,rate=44100,input=True,frames_per_buffer=2**11) :
        self.__streams.clear()
        self.__format = format
        self.__channels = channels
        self.__rate = rate
        self.__chunk = frames_per_buffer

        for i in range(self.__MIKE_COUNT) :
            stream = self.__audio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=i,
                frames_per_buffer=frames_per_buffer
            )

            self.__streams.append(stream)
        return

    # If you want to devide stereo_sound to left_channel and right_channel,
    # devide_stereo = True
    def write_sound(self, path, suffix='.wav', devide_stereo=False) :
        i = 0
        if devide_stereo == True :
            if self.__channels == 2 :
                frames_r = []
                frames_l = []

                for frames in self.__sound_frames :
                    for frame in frames :
                        frame_l = audioop.tomono(frame, 2, 1, 0)
                        frames_l.append(frame_l)
                        frame_r = audioop.tomono(frame, 2, 0, 1)
                        frames_r.append(frame_r)

                    filename = path + str(i)

                    self.__write_soundfile(frames_r, filename + '_r' + suffix, 1)
                    self.__write_soundfile(frames_l, filename + '_l' + suffix, 1)
                    i = i + 1
                    frames_l.clear()
                    frames_r.clear()

            else :
                print('Recording Sound is not Stereo Sound.')
                print('So, Write only Monaural Sound')
            
        i = 0
        for frame in self.__sound_frames :
            filename = path + str(i) + suffix
            self.__write_soundfile(frame, filename)
            i = i + 1

        if self.__is_write_log is False :
            self.__write_log(path)
            self.__is_write_log = True

        
        return

    # If you want to devide_stereo or change_channels, 
    # select channel/ 1:Monaural, 2:Stereo
    def __write_soundfile(self, frames, filename, channels=0) :
        if channels == 0 :
            channels = self.__channels

        wave_file = wave.open(filename, "w")
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(self.__audio.get_sample_size(self.__format))
        wave_file.setframerate(self.__rate)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        return

    def write_video(self, path, sight_path, suffix='.avi', fps=30.0) :
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        i = 0

        for frame in self.__video_frames :
            out = cv2.VideoWriter(path + str(i) + suffix, fourcc, fps, (640, 480))
            count = 0
            for f in frame :
                count = count + 1
                out.write(f)
            out.release()
            print(str(i) + ':' + str(count))
            i = i + 1

        i = 0
        for frame in self.__base_sight_frames :
            out = cv2.VideoWriter(sight_path + str(i) + suffix, fourcc, fps, (640, 480))
            count = 0
            for f in frame :
                count = count + 1
                out.write(f)
            out.release()
            print(str(i) + ':' + str(count))
            i = i + 1

        if self.__is_write_log is False :
            self.__write_log(path)
            self.__is_write_log = True
        return

    def __write_log(self, path) :
        f = open(path + "log.txt", "w")
        f.write(self.__logs)
        f.close()

        f = open(path + 'talk.txt', "w")
        f.write('1:\n2:\n3:\n4:\n5:\n6:\n')
        f.close()

    # Finally, you must call it
    def relese(self) :
        for cap in self.__caps :
            cap.relese()

        cv2.destroyAllWindows()