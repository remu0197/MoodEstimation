import glob, os, csv
import cv2
import pandas as pd
import numpy as np
import wave
import pydub
from pydub import AudioSegment
import shutil

def main() :
    DATASET_PATHES = glob.glob('../../../experiment/data/*')

    for path in DATASET_PATHES :
        dir_name = os.path.basename(path)
        if dir_name.isdecimal() is False :
            continue

        audio_pathes = glob.glob(path + '/*.wav')
        # video_pathes = glob.glob(path + '/*.avi')
        if len(audio_pathes) == 0 or len(video_pathes) == 0 :
            print('Lack of dataset in: ' + dir_name)
            continue
        
        print(dir_name)
        for audio in audio_pathes :
            get_fps(audio, '')

        continue

        sections_path = path + '/sections.csv'
        if os.path.exists(sections_path) is False:
            continue

        sections = pd.read_csv(sections_path)
        for _, section in sections.iterrow() :
            start, end = section.start, section.end

def extract_sound(path, file_index=3) :
    dir_name = os.path.basename(path)
    read_path = path + '/0.wav'
    write_path = '../../../experiment/edit/separation_sound/' + dir_name

    if not os.path.exists(read_path) :
        print('not exist : ' + read_path)
        return 

    if not os.path.exists(write_path) :
        os.mkdir(write_path)
    else :
        print('Is exist: ' + write_path)
        return

    sections_path = path + '/sections.csv'
    if os.path.exists(sections_path) is False:
        return False

    sections = pd.read_csv(sections_path)
    for i, section in sections.iterrows() :
        start, end = section.start, section.end

        edit_start = (int(start * 30) + 1) / 30 * 1000
        edit_end = int(end * 30) / 30 * 1000

        base_sound = AudioSegment.from_file(read_path, format="wav")
        edit_sound =  base_sound[edit_start:edit_end]
        write_audio_path = write_path + '/Section' + str(i) + '.wav'
        print("Log: " + write_audio_path)
        edit_sound.export(write_audio_path, format="wav")

def extract_video(base_path) :
    dir_name = os.path.basename(base_path)
    write_path = '../../../experiment/edit/separation_video/' + dir_name 
    log_path = '../../../experiment/edit/separation_video/short_videos.csv'   
    video_pathes = glob.glob(base_path + '/[0-9].avi')
    sections_path = base_path + '/sections.csv'

    if len(video_pathes) < 3 or os.path.exists(sections_path) is False:
        print('Lack of dataset in: ' + dir_name)
        return
    if not os.path.exists(write_path) :
        os.mkdir(write_path)
    else :
        print('Is exist: ' + write_path)
        return

    sections = pd.read_csv(sections_path)
    video_frames = []
    for _, section in sections.iterrows() :
        start, end = section.start, section.end

        edit_start = (int(start * 30) + 1)
        edit_end = int(end * 30)
        video_frames.append([edit_start, edit_end])

    print(len(video_pathes))
    for i, path in enumerate(video_pathes) :
        cap = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) != 300 :
            print('Lack of frame count') 
            with open(log_path, "a") as f :
                writer = csv.writer(f)
                writer.writerow([
                    path,
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)])
        else :
            print('ok')

        for j, frames in enumerate(video_frames) :
            print(path + ' : ' + str(frames[0]) + ' - ' + str(frames[1]))

            out = cv2.VideoWriter(
                write_path + '/' + str(i) + '_' + str(j) + '.avi',
                fourcc,
                fps,
                (width, height))

            for f in range(frames[0], frames[1]) :
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                _, frame = cap.read()
                out.write(frame)

            out.release()

    return True

def extract_sections(datapath) :
    sections_path = datapath + '/sections.csv'
    fps = 44100
    min_section_len = fps * 20
    section_count = 0
    section_index = 1

    if os.path.exists(sections_path) is True:
        # print('Sections is exist at : ' + datapath)
        return False

    print('TARGET:  ' + datapath)

    x_base = read_wav_buffer(datapath + '/0_r.wav', '')
    x_edit = read_wav_buffer(datapath + '/Section' + str(section_index) + '.wav')
    sections = []
    start = 0
    is_researched = False

    for i in range(len(x_base)) :
        if x_base[i] == x_edit[section_count] :
            section_count = section_count + 1
        else :
            section_count = 0
            start = i + 1

        if section_count == len(x_edit) - 1 :
            section = [start / fps, i / fps, (i - start) / fps]
            print(section)
            sections.append(section)
            section_count = 0
            section_index = section_index + 1
            if section_index >= 5 :
                is_researched = True
                break
            x_edit = read_wav_buffer(datapath + '/Section' + str(section_index) + '.wav')

    if is_researched is True :
        print('COMPLETE : ' + sections_path)
        with open(sections_path, "w") as f :
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['start', 'end', 'length'])
            writer.writerows(sections)
    else :
        print('FAIL')

def read_wav_buffer(path, log='TARGET:  ') :
    wr = wave.open(path)
    data = wr.readframes(wr.getnframes())
    wr.close()
    
    return np.frombuffer(data, dtype="int16") / float((2^15))

def get_fps(audio_path, video_path) :
    # cap = cv2.VideoCapture(video_path)
    # vfps = float(1 / cap.get(cv2.CAP_PROP_FPS))
    # cap.release()

    wf = wave.open(audio_path)
    print(wf.getnframes() / wf.getframerate())
    afps = float(1 / wf.getframerate())
    # afps = float(1 / 44100)

    return afps

def change_dir() :
    pathes = glob.glob('../../../experiment/edit/separation_video/*')
    
    for path in pathes :
        dir_name = os.path.basename(path)
        if not os.path.isdir(path) :
            continue
        
        videos = glob.glob(path + '/*.avi') 
        for video in videos :
            filename = os.path.basename(video)
            if filename[0] is '2' :
                continue
            
            write_path = '../../../experiment/edit/movie/visual/' + dir_name + '_' + filename
            if not os.path.exists(write_path) :
                shutil.copyfile(video, write_path)

    pathes = glob.glob('../../../experiment/edit/separation_sound/*')

    for path in pathes :
        dir_name = os.path.basename(path)
        if not os.path.isdir(path) :
            continue
        
        sounds = glob.glob(path + '/*.wav') 
        for sound in sounds :
            filename = os.path.basename(sound)
            
            write_path = '../../../experiment/edit/movie/audio/' + dir_name + '_0_' + filename.lstrip('Section')
            if not os.path.exists(write_path) :
                shutil.copyfile(sound, write_path)
            write_path = '../../../experiment/edit/movie/audio/' + dir_name + '_1_' + filename.lstrip('Section')
            if not os.path.exists(write_path) :
                shutil.copyfile(sound, write_path)

    # shutil.copyfile("./test1/test1.txt", "./test2.txt")

if __name__ == "__main__":

    extract_sound('../../../experiment/data/1812201207')
    extract_video('../../../experiment/data/1812201207')
    change_dir()
    # extract_sections('../../../experiment/data/1812141620')
    # extract_video('../../../experiment/data/1812201207')

    # change_dir()
    
    # dirs = glob.glob('../../../experiment/data/*')
    # for i, d in enumerate(dirs) :
    #     # extract_sections(d)
    #     # continue
    #     # extract_video(d)
    #     extract_sound(d)


    # extract_sound(
    #     path='../../../experiment/data/19121312541', 
    #     file_index='3')

    # 1812111447 1812181292 1812181337 1812201343 1905221218