import glob, csv, os, sys, shutil
import pandas as pd
from pydub import AudioSegment

class SoundCreater:
  def __init__(self):
    self.__DATASET_DIR = "../data/dataset/"
    
    self.__id_dirlist = []
    pathes = glob.glob(self.__DATASET_DIR + "*")
    for path in pathes:
      basename = os.path.basename(path)
      self.__id_dirlist.append(basename)

  def __extract_state(self, id_dir, export_dirpath, state_num, is_update=False):
    id_list, list_pathes = id_dir.split("_"), []
    group_dirpath = export_dirpath + id_dir + "/"

    if not os.path.exists(group_dirpath):
      print(group_dirpath)
      os.mkdir(group_dirpath)
    elif is_update is False:
      print(" SKIP owing to is_update is False")
      return False
    else:
      shutil.rmtree(group_dirpath)
      os.mkdir(group_dirpath)

    for i, id in enumerate(id_list):
      list_pathes = glob.glob("../data/state_list/" + str(int(id)) + "_*.csv")

      for j, list_path in enumerate(list_pathes):
        datapath = self.__DATASET_DIR + id_dir + "/" + str(i) + "_" + str(j) + ".wav"

        print("TARGET : " + datapath)          
        if not os.path.exists(datapath):
          print(" ERROR : Not Exist Data")
          continue

        base_sound = AudioSegment.from_file(datapath, format="wav")
        edit_sounds = []

        with open(list_path, 'r') as f:
          reader = csv.reader(f)
          past_time = 0
          for row in reader:
            if(len(row)) == 0:
              continue

            value = int(float(row[1]) * 1000)
            if value == 0:
              continue

            if int(row[0]) == state_num:
              edit_sounds.append(base_sound[past_time:past_time+value])
            past_time += value

        if len(edit_sounds) == 0:
          continue
        
        for k, edit_sound in enumerate(edit_sounds):
          export_path = group_dirpath + str(i) + "_" + str(j) + "_" + str(k) + ".wav"
          print(" EXPORT  : " + export_path)
          edit_sound.export(export_path, format="wav")

  def extract_state(self, state_num, is_update=False):
    export_dir = "../data/state_sound/default/" + str(state_num) + "/"
    if not os.path.exists(export_dir):
      os.mkdir(export_dir)
    
    for id_dir in self.__id_dirlist:
      self.__extract_state(id_dir, export_dir, state_num, is_update=is_update)
      
  def concatinate_state(self, state_num, is_update=False):
    import_dir = "../data/state_sound/default/" + str(state_num) + "/"
    export_dir = "../data/state_sound/concatinate/"+ str(state_num) + "/"

    if not os.path.exists(export_dir):
      os.mkdir(export_dir)

    for id_dir in self.__id_dirlist:
      if not os.path.exists(import_dir + id_dir):
        self.__extract_state(id_dir, import_dir, state_num, is_update=False)

      print("CONCSTINATE_TARGET: " + import_dir + id_dir)

      id_list = id_dir.split("_")

      if not os.path.exists(export_dir + id_dir):
        os.mkdir(export_dir + id_dir)

      for i in range(len(id_list)):
        for j in range(4):
          export_path = export_dir + id_dir + "/" + str(i) + "_" + str(j) + ".wav"
          if os.path.exists(export_path) and not is_update:
            print(" SKIP: " + export_path)
            continue

          base_sounds = []
          base_pathes = glob.glob(import_dir + id_dir + "/" + str(i) + "_" + str(j) + "_*.wav")
          for base_path in base_pathes:
            base_sound = AudioSegment.from_file(base_path, format="wav")
            base_sounds.append(base_sound)

          if len(base_sounds) == 0:
            print(" ERROR: No Section of " + str(state_num))
            continue

          edit_sound = base_sounds[0]
          for base_sound in base_sounds[1:]:
            edit_sound = edit_sound + base_sound

          print(" EXPORT: " + export_path)
          edit_sound.export(export_path, format="wav")

  #   # group_dirpathes = glob.glob(import_dir + "/*")
  #   # for group in group_dirpathes:
  #   #   id_list = group.split("_")
  #   #   basesounds = []
      
  #   #   for i in range(len(id_list)):
  #   #     for j in range(4):


  #   # dataset_dir = "../data/dataset/"
  #   # result_dir = "../data/sound_state" + str(state_num) + "/"

  #   # pathes = glob.glob(dataset_dir + "*")
  #   # for path in pathes:
  #   #   basename = os.path.splitext(os.path.basename(path))[0]
  #   #   id_list = basename.split('_')

  #   #   for i, id in enumerate(id_list):
  #   #     states_pathes = glob.glob("../data/state_list/" + str(int(id)) + "_*.csv")
        
  #   #     for j, states_path in enumerate(states_pathes):
  #   #       states = []
  #   #       sound_path = dataset_dir + basename + "/" + str(i) + "_" + str(j) + ".wav"
  #   #       if not os.path.exists(sound_path):
  #   #         continue

  #   #       base_sound = AudioSegment.from_file(sound_path, format="wav")
  #   #       past_time = 0

  #   #       with open(states_path, 'r') as f:
  #   #         reader = csv.reader(f)
  #   #         for row in reader:
  #   #           if len(row) == 0:
  #   #             continue

  #   #           value = int(float(row[1]) * 1000)
  #   #           if int(row[0]) == state_num:
  #   #             states.append(base_sound[past_time:past_time + value])
  #   #           past_time = past_time + value

  #   #       if len(states) is 0:
  #   #         continue

  #   #       edit = states[0]
  #   #       for state in states[1:]:
  #   #         edit = edit + state

  #   #       export_dir = result_dir + basename + "/"
  #   #       if not os.path.exists(export_dir):
  #   #         os.mkdir(export_dir)

  #   #       export_path = export_dir + str(i) + "_" + str(j) + ".wav"
  #   #       print("CREATE:  " + export_path)        
  #   #       edit.export(export_path, format="wav")      

  # def concatinate_sstrain(self, state_num):
  #   result_dir = "../data/ss_train/"
  #   dataset_dir = "../data/dataset/"

  #   pathes = glob.glob(dataset_dir + "*")
  #   for path in pathes:
  #     basename = os.path.splitext(os.path.basename(path))[0]
  #     id_list = basename.split('_')

  #     for i, id in enumerate(id_list):
  #       states = []
  #       states_pathes = glob.glob("../data/state_list/" + str(int(id)) + "_*.csv")
        
  #       for j, states_path in enumerate(states_pathes):
  #         sound_path = dataset_dir + basename + "/" + str(i) + "_" + str(j) + ".wav"
  #         if not os.path.exists(sound_path):
  #           continue

  #         base_sound = AudioSegment.from_file(sound_path, format="wav")
  #         past_time = 0
  #         with open(states_path, 'r') as f:
  #           reader = csv.reader(f)
  #           for row in reader:
  #             if len(row) == 0:
  #               continue

  #             value = int(float(row[1]) * 1000)
  #             if int(row[0]) == state_num:
  #               states.append(base_sound[past_time:past_time + value])
  #             past_time = past_time + value

  #       if len(states) is 0:
  #         continue

  #       edit = states[0]
  #       for state in states[1:]:
  #         edit = edit + state

  #       edit_path = result_dir + id + "_" + str(state_num) + ".wav"
  #       print("CREATE:  " + edit_path)        
  #       edit.export(edit_path, format="wav")

  #   # pathes = glob.glob("../data/state_list/*")
  #   # for path in pathes :
  #   #   past_time = 0.0
  #   #   with open(path, 'r') as f:
  #   #     reader = csv.reader(f)
  #   #     state_list = []
  #   #     for row in reader:
  #   #       if len(row) == 0:
  #   #         continue
  #   #       if int(row[0]) == state_num:
  #   #         state_list.append([
  #   #           int(past_time*1000), 
  #   #           int(past_time + float(row[1])*1000)
  #   #         ])
  #   #       past_time = past_time + float(row[1])
        
  #   #     base = AudioSegment.from_file(dataset_dir + )
  #   #     edit = base[state_list[0][0]:state_list[0][1]]
  #   #     for state in state_list[1:]:
  #   #       edit = edit + base[state[0]:state[1]]


if __name__ == "__main__":
    SC = SoundCreater()
    for i in range(1, 2):
     SC.extract_state(i, is_update=False)
     SC.concatinate_state(i, is_update=False)


## たぶん音声の終了時間がずれてーら
  # 43_2
  # 35_3
  # 35_1
  # 31_2
  # 27_1
  # 28_1
  # 16_1
  # 15_1
  # 11_1