import glob, csv, os
import pandas as pd
from pydub import AudioSegment

class SoundCreater:
  def concatinate_one_states(self, state_num):
    dataset_dir = "../data/dataset/"
    result_dir = "../data/sound_state" + str(state_num) + "/"

    pathes = glob.glob(dataset_dir + "*")
    for path in pathes:
      basename = os.path.splitext(os.path.basename(path))[0]
      id_list = basename.split('_')

      for i, id in enumerate(id_list):
        states_pathes = glob.glob("../data/state_list/" + str(int(id)) + "_*.csv")
        
        for j, states_path in enumerate(states_pathes):
          states = []
          sound_path = dataset_dir + basename + "/" + str(i) + "_" + str(j) + ".wav"
          if not os.path.exists(sound_path):
            continue

          base_sound = AudioSegment.from_file(sound_path, format="wav")
          past_time = 0

          with open(states_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
              if len(row) == 0:
                continue

              value = int(float(row[1]) * 1000)
              if int(row[0]) == state_num:
                states.append(base_sound[past_time:past_time + value])
              past_time = past_time + value

          if len(states) is 0:
            continue

          edit = states[0]
          for state in states[1:]:
            edit = edit + state

          export_dir = result_dir + basename + "/"
          if not os.path.exists(export_dir):
            os.mkdir(export_dir)

          export_path = export_dir + str(i) + "_" + str(j) + ".wav"
          print("CREATE:  " + export_path)        
          edit.export(export_path, format="wav")      

  def concatinate_sstrain(self, state_num):
    result_dir = "../data/ss_train/"
    dataset_dir = "../data/dataset/"

    pathes = glob.glob(dataset_dir + "*")
    for path in pathes:
      basename = os.path.splitext(os.path.basename(path))[0]
      id_list = basename.split('_')

      for i, id in enumerate(id_list):
        states = []
        states_pathes = glob.glob("../data/state_list/" + str(int(id)) + "_*.csv")
        
        for j, states_path in enumerate(states_pathes):
          sound_path = dataset_dir + basename + "/" + str(i) + "_" + str(j) + ".wav"
          if not os.path.exists(sound_path):
            continue

          base_sound = AudioSegment.from_file(sound_path, format="wav")
          past_time = 0
          with open(states_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
              if len(row) == 0:
                continue

              value = int(float(row[1]) * 1000)
              if int(row[0]) == state_num:
                states.append(base_sound[past_time:past_time + value])
              past_time = past_time + value

        if len(states) is 0:
          continue

        edit = states[0]
        for state in states[1:]:
          edit = edit + state

        edit_path = result_dir + id + "_" + str(state_num) + ".wav"
        print("CREATE:  " + edit_path)        
        edit.export(edit_path, format="wav")

    # pathes = glob.glob("../data/state_list/*")
    # for path in pathes :
    #   past_time = 0.0
    #   with open(path, 'r') as f:
    #     reader = csv.reader(f)
    #     state_list = []
    #     for row in reader:
    #       if len(row) == 0:
    #         continue
    #       if int(row[0]) == state_num:
    #         state_list.append([
    #           int(past_time*1000), 
    #           int(past_time + float(row[1])*1000)
    #         ])
    #       past_time = past_time + float(row[1])
        
    #     base = AudioSegment.from_file(dataset_dir + )
    #     edit = base[state_list[0][0]:state_list[0][1]]
    #     for state in state_list[1:]:
    #       edit = edit + base[state[0]:state[1]]


if __name__ == "__main__":
    SC = SoundCreater()
    SC.concatinate_sstrain(1)
    SC.concatinate_sstrain(2)