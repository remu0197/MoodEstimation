import glob, subprocess, csv, os, sys
from scipy.io import arff
import numpy as np

class OpenSMILE:
  def __init__(self):
    self.__CONFIG_PATH = "C:/Program Files/opensmile-2.3.0/config/IS10_paraling.conf"
    
  def solo_opensmile(self, sound_dir):
    group_dirs = glob.glob(sound_dir + "/*")
    
    for group_dir in group_dirs:
      group_id = os.path.basename(group_dir)
      subject_id = group_id.split("_")
      
      for i, id in enumerate(subject_id):
        export_dir = "../data/solo_opensmile/arff/"
        # if not os.path.exists(export_dir):
        #   os.mkdir(export_dir)

        # csv_dir = "../data/solo_opensmile/csv/" + id
        # if not os.path.exists(csv_dir):
        #   os.mkdir(csv_dir)

        for j in range(4):
          sound_pathes = glob.glob(group_dir + "/" + str(i) + "_" + str(j) + "_*.wav")

          for path in sound_pathes:
            subprocess.call([
              "SMILExtract_Release", 
              "-C", self.__CONFIG_PATH, 
              "-I", path, 
              "-O", export_dir + "/all.arff",
              "-instname", path.replace('\\', '/'),
            ])

  def sound_to_arff(self, sound_dir, format="wav"):
    # sound_pathes = glob.glob(sound_dir + "/**/*." + format)
    sound_pathes = glob.glob(sound_dir + "/**/extract_sound/*." + format)
    export_path = "../data/IS10_CD.arff"

    if os.path.exists(export_path):
      os.remove(export_path)

    for path in sound_pathes:
      print(path)
      subprocess.call([
        "SMILExtract_Release", 
        "-C", self.__CONFIG_PATH, 
        "-I", path, 
        "-O", export_path,
        "-instname", path,
      ])

  def arff_to_csv(self, arff_path, csv_path):
    dataset, meta = arff.loadarff(arff_path)
    ds = np.asarray(dataset.tolist(), dtype=np.float32)
    np.savetxt(csv_path, ds, delimiter=',', fmt='.6e')

if __name__ == "__main__":
    OS = OpenSMILE()
    OS.solo_opensmile("../../processing/data/state_sound/default/1/")
    # OS.arff_to_csv("../data/IS10_CD.arff", "../data/IS10_CD.csv")
    # OS.sound_to_arff("../../processing/data/consortium/edit/")
    # OS.sound_to_arff("../../processing/data/state_sound/concatinate/1/")
    # OS.arff_to_array()


    # C:\Users\remu\Documents\Labo\Mood\opensmile-2.3.0\config

    # SMILExtract_Release -C .\config\IS11_speaker_state.conf -I .\example-audio\opensmile.wav -O sample_11.csv 

    # SMILExtract_Release -C .\config\IS10_paraling.conf -I .\example-audio\opensmile.wav -O sample.csv                                   