import glob, subprocess, csv, os
from scipy.io import arff
import numpy as np

class OpenSMILE:
  def sound_to_arff(self, sound_dir, format="wav"):
    CONFIG_PATH = "C:/Users/remu/Documents/Labo/Mood/opensmile-2.3.0/config/IS10_paraling.conf"
    sound_pathes = glob.glob(sound_dir + "/**/*." + format)
    export_path = "../data/IS10.arff"

    if os.path.exists(export_path):
      os.remove(export_path)

    for path in sound_pathes:
      subprocess.call([
        "SMILExtract_Release", 
        "-C", CONFIG_PATH, 
        "-I", path, 
        "-O", export_path,
        "-instname", path,
      ])

if __name__ == "__main__":
    OS = OpenSMILE()
    # OS.sound_to_arff("../../processing/data/sound_state1")
    # OS.arff_to_array()


    # C:\Users\remu\Documents\Labo\Mood\opensmile-2.3.0\config

    # SMILExtract_Release -C .\config\IS11_speaker_state.conf -I .\example-audio\opensmile.wav -O sample_11.csv 

    # SMILExtract_Release -C .\config\IS10_paraling.conf -I .\example-audio\opensmile.wav -O sample.csv                                   