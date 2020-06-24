import os, glob, cv2

class VideoEditor :
    # def __init__(self) :


    def video_hcombine(self, path_l, path_r, out_path) :
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        cap_l, cap_r = cv2.VideoCapture(path_l), cv2.VideoCapture(path_r)
        video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (1280, 480))

        while cap_l.isOpened() : # and cap_r.isOpened() :
            _, img_l = cap_l.read()
            _, img_r = cap_r.read()

            if img_l is None or img_r is None :
                cap_l.release()
                cap_r.release()
                break

            img_combine = self.__hcombine(img_l, img_r)
            video_writer.write(img_combine)

        video_writer.release()
        print('Complete writiing: ' + out_path)

        return True

    def __hcombine(self, img_l, img_r, is_color=True) :
        h_r, w_r, _ = img_r.shape[:2+int(is_color)]
        h_l, w_l, _ = img_l.shape[:2+int(is_color)]

        # Fit to bigger one
        if h_r < h_l :
            h_r = h_l
            w_r = int((h_l / h_r) * w_l)
            img_r = cv2.resize(img_r, (w_r, h_r))
        elif h_r > h_l :
            h_l = h_r
            w_l = int((h_r / h_l) * w_r)
            img_l = cv2.resize(img_l, (w_l, h_l))

        img = cv2.hconcat([img_l, img_r])

        # cv2.imshow("Frame", img)
        cv2.waitKey(1) & 0xFF

        return img

if __name__ == "__main__":
    result_dir = '../data/'
    pathes = glob.glob('../data/*')
    dirs = os.listdir('../data/')
    editor = VideoEditor()

    for i, path in enumerate(pathes) :
        video_pathes = glob.glob(path + '/[0-9].avi')
        print(video_pathes)
        out_path = result_dir + dirs[i] + '/2.avi'
        # if not os.path.isdir(out_path):
        #     os.mkdir(out_path)

        # out_path = out_path + '0_1.avi'

        if len(video_pathes) >= 2 :
            video_path_l = video_pathes[0].replace('\\', '/')
            video_path_r = video_pathes[1].replace('\\', '/')
            editor.video_hcombine(video_path_l, video_path_r, out_path)
        else :
            print('Lack of videos at: ' + path)
    
    # editor = VideoEditor()
    # editor.video_hcombine(path_l='./0.avi', path_r='./1.avi', out_path='../edit/hcombine/1812111310/0-1.avi')

    cap = cv2.VideoCapture('./0.avi')

    while cap.isOpened() :
        _, frame = cap.read()

        if frame is None :
            cap.release()
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF