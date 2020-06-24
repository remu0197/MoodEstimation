# import cv2
# import time

# class CapturingManager :
#     def __init__(self, device_count, recording_second, path):
#         self.__DEVICE_COUNT = device_count
#         self.__RECORDING_SECOND = recording_second
#         self.__caps = []

#         for i in range(self.__DEVICE_COUNT):
#             self.__caps.append(cv2.VideoCapture(i))

#         self.__FOURCC = cv2.VideoWriter_fourcc(*'XVID')
#         self.__outs = []
        
#         for i in range(device_count):
#             self.__outs.append(cv2.VideoWriter(path + '/' + str(i) + '.avi', self.__FOURCC, 30.0, (1280, 1080)))

#     #def Capturing(self):
#         #TODO

#     def capturing_with_mike(self, index):
#         ret, frame = self.__caps[index].read()

#         frame = frame[::-1,:]
#         if ret == False:
#             return False
#         else :
#             frame = cv2.flip(frame, 0)

#             self.__outs[index].write(frame)
#             cv2.imshow(frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 return False

#     def release(self) :
#         for cap in self.__caps :
#             cap.release()
#         for out in self.__outs :
#             out.release()
#         cv2.destroyAllWindows()

#     def get_device_count(self):
#         return self.__DEVICE_COUNT



import cv2
import time

class CapturingManager :
    def __init__(self, path, recording_second) :
        self.__RECORDING_SECOND = recording_second

    def capture(self, path, index) :
        # この辺メンバで持たせると[can't pickle -]ってエラー
        # 準備クラスを別で作ってとか

        cap = cv2.VideoCapture(index)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(path + '/' + str(index) + '.avi', fourcc, 30.0, (640, 480))

        frames = []

        # mainで定義して整合性をとる
        # 時間がおかしい原因はcap.read()初回起動時のラグ（約１秒）
        # print('start:' + str(current_time))
        ret, frame = cap.read()
        print('start')
        current_time = time.time()
    
        while time.time() - current_time < self.__RECORDING_SECOND:
            print(str(index) + '.' + str(time.time()))
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            frames.append(frame)

        print('end')

        for frame in frames :
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # # cv2.imshow('frame', frame)
                

def func(second, path) :
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path + '/' + '0.avi', fourcc, 30.0, (640, 480))

    current_time = time.time()

    while time.time() - current_time < second:
        ret, frame = cap.read()

        frame = frame[::-1, :]
        if ret == True:
            frame = cv2.flip(frame, 0)

            cv2.imshow(frame)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break
        else :
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
