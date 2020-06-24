from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
import glob
import csv

class headPose :
    def __init__(self, datapath) :
        self.__videos = []
        self.__landmarks_path = "../resource/shape_predictor_68_face_landmarks.dat"
        self.__pathes = glob.glob(datapath + '/**/*.avi')

    def test(self) :
        # for i, path in enumerate(self.__pathes) :
        #     result = self.get_head_vector(path, i)
        path = '../../../experiment/data/1812111310/1.avi'
        # self.show_head_pose(path)
        self.get_head_vector(path)
        

    def show_head_pose(self, path) :
        cap = cv2.VideoCapture(path)

        while cap.isOpened() :
            _, frame = cap.read()          
            if frame is None :
                cap.release()
                break

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            height, width = frame.shape[:2]
            b, g, r = cv2.split(frame)
            zeros = np.zeros((height, width), frame.dtype)
            frame_r = cv2.merge((zeros, zeros, r))
            
            cv2.imshow("Frame", frame_r)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break


        return True

    def extract_foreground(self) :
        #thresh_hsv.py
        # -*- coding: utf-8 -*-
        # thresh = 0
        # max_val = 255
        # thresholdType = cv2.THRESH_BINARY
        # #トラックバーで、しきい値を変更
        # def changethresh(pos):
        #     global thresh
        #     thresh = pos
        # filename = sys.argv[1]
        # #画像を読み込む
        # img = cv2.imread(filename)
        # #BGRをHSVに変換
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # #HSVに分解
        # h_img, s_img, v_img = cv2.split(hsv)
        # #ウィンドウの名前を設定
        # cv2.namedWindow("img")
        # cv2.namedWindow("thresh")
        # #トラックバーのコールバック関数の設定
        # cv2.createTrackbar("trackbar", "thresh", 0, 255, changethresh)
        # while(1):
        #     cv2.imshow("img", img)
        #     _, thresh_img = cv2.threshold(s_img, thresh, max_val, thresholdType)
        #     cv2.imshow("thresh", thresh_img)
        #     k = cv2.waitKey(1)
        #     #Escキーを押すと終了
        #     if k == 27:
        #         break
        #     #sを押すと結果を保存
        #     if k == ord("s"):
        #         result = cv2.merge(cv2.split(img) + [thresh_img])
        #         cv2.imwrite(filename[:filename.rfind(".")] + "_result.png", result)
        #         break

        return True

    def get_head_vector(self, path, fileIndex=0) :
        print('Current target:' + path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.__landmarks_path)

        cap = cv2.VideoCapture(path)

        frame_count = 0
        lost_count = 0
        result = []

        while cap.isOpened() :
            frame_count += 1
            _, frame = cap.read()
            if frame is None :
                cap.release()
                break
            frame = frame[100:370, 240:600]
            frame = cv2.resize(frame, dsize=(1200, 900))
            # frame = imutils.resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if len(rects) < 1 :
                print(' Error: No.' + str(frame_count) + ' is not proper state in ' + path)
                lost_count += 1
                result.append([])
            # else:
            #     print(frame_count)

            image_points = None
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                        tuple(shape[48]), tuple(shape[54])])

                for (x, y) in image_points:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                        tuple(shape[48]), tuple(shape[54])], dtype='double')

            if len(rects) > 0:
                cv2.putText(frame, "detected", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

                model_points = np.array([
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0)
                ])

                size = frame.shape

                focal_length = size[1]
                center = (size[1] // 2, size[0] // 2)

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype='double')

                dist_coeffs = np.zeros((4, 1))

                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
                mat = np.hstack((rotation_matrix, translation_vector))
                # homogeneous transformation matrix (projection matrix)

                (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
                # yaw = eulerAngles[1]
                # pitch = eulerAngles[0]
                # roll = eulerAngles[2]
                
                result.append(eulerAngles)
                # print(len(result))

                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)                

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # self.write_csv(result, '../data/head/', fileIndex)


        return True

    def write_csv(self, data, path, fileIndex) :
        with open(path + str(fileIndex) + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(data)


    # def hoge(self) :
        # # video = "../resource/headPose.jpg"

        # print("[INFO] loading facial landmark predictor...")
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(path)

        # # initialize the video stream and allow the cammera sensor to warmup
        # # print("[INFO] camera sensor warming up...")
        # # vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
        # cap = cv2.VideoCapture(video)

        # while cap.isOpened():
        #     # frame = vs.read()
        #     _, frame = cap.read()
        #     # frame = cv2.imread(video)
        #     frame = imutils.resize(frame, width=800)
        #     frame = self.gamma_corection(frame, gamma=1.0)
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #     rects = detector(gray, 0)
        #     print(len(rects))

        #     image_points = None
        #     for rect in rects:
        #         shape = predictor(gray, rect)
        #         shape = face_utils.shape_to_np(shape)
        #         print(len(shape))

        #         image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
        #                                 tuple(shape[48]), tuple(shape[54])])

        #         for (x, y) in image_points:
        #             cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        #         image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
        #                                 tuple(shape[48]), tuple(shape[54])], dtype='double')

        #     if len(rects) > 0:
        #         cv2.putText(frame, "detected", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

        #         model_points = np.array([
        #             (0.0, 0.0, 0.0),
        #             (0.0, -330.0, -65.0),
        #             (-225.0, 170.0, -135.0),
        #             (225.0, 170.0, -135.0),
        #             (-150.0, -150.0, -125.0),
        #             (150.0, -150.0, -125.0)
        #         ])

        #         size = frame.shape

        #         focal_length = size[1]
        #         center = (size[1] // 2, size[0] // 2)

        #         camera_matrix = np.array([
        #             [focal_length, 0, center[0]],
        #             [0, focal_length, center[1]],
        #             [0, 0, 1]
        #         ], dtype='double')

        #         dist_coeffs = np.zeros((4, 1))

        #         (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
        #                                                                     dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #         (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        #         mat = np.hstack((rotation_matrix, translation_vector))
        #         # homogeneous transformation matrix (projection matrix)

        #         (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        #         yaw = eulerAngles[1]
        #         pitch = eulerAngles[0]
        #         roll = eulerAngles[2]

        #         cv2.putText(frame, 'yaw' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        #         cv2.putText(frame, 'pitch' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        #         cv2.putText(frame, 'roll' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        #         (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
        #                                                         translation_vector, camera_matrix, dist_coeffs)

        #         for p in image_points:
        #             cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        #         p1 = (int(image_points[0][0]), int(image_points[0][1]))
        #         p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        #         cv2.line(frame, p1, p2, (255, 0, 0), 2)

        #     # show the frame
        #     cv2.imshow("Frame", frame)
        #     key = cv2.waitKey(1) & 0xFF

        #     # if the `q` key was pressed, break from the loop
        #     if key == ord("q"):
        #         break

        # cv2.destroyAllWindows()
        # # vs.stop()

    def gamma_corection(self, frame, gamma) :
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256) :
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        return cv2.LUT(frame, lookUpTable)

    def adjust_color(self, frame, saturation, brightness) :
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame_hsv[:,:,(1)] = frame_hsv[:,:,(1)] * saturation
        frame_hsv[:,:,(2)] = frame_hsv[:,:,(2)] * brightness

        return cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    x = headPose('../../../experiment/data')
    x.test()