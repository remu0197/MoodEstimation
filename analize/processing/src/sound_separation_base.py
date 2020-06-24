import numpy as np
import librosa
import scipy.io.wavfile as wav
from pydub import AudioSegment
import glob, os, csv, datetime
import concurrent.futures
import multiprocessing

class SoundSeparation :
  def double_to_single(self,d_path,l_path,r_path):
    sound = AudioSegment.from_file(d_path, format="wav")
    samples = np.array(sound.get_array_of_samples())

    if sound.channels is 1:
      print(d_path + " is monoral")
      l_samples = samples
      r_samples = samples
    else:
      l_samples = samples[0:len(samples):2]
      r_samples = samples[0:len(samples):2]

    if not os.path.exists(l_path):
      self.__export_wav(
        path=l_path,
        samples=l_samples,
        width=sound.sample_width,
        frame_rate=sound.frame_rate,
        channels=1,
      )

    if not os.path.exists(r_path):
      self.__export_wav(
        path=r_path,
        samples=r_samples,
        width=sound.sample_width,
        frame_rate=sound.frame_rate,
        channels=1,
      )    

  def separate(self,R_t,n_iter_t,R,n_iter,predict_pathes, train_path, r_pathes):
    print("TRAINING: " + train_path)

    y_train, _ = librosa.load(train_path)
    s_train = librosa.stft(y_train)

    nmf = self.__nmf(np.abs(s_train), R=R_t, n_iter=n_iter_t)
    base = nmf[0]

    for i, predict in enumerate(predict_pathes):
      print("PRIDICT: " + predict)
      y_b, sr_b = librosa.load(predict)
      s_b = librosa.stft(y_b)

      ssnmf = self.__ssnmf(np.abs(s_b), R=R, F=base, n_iter=n_iter)    
      result = librosa.istft((np.dot(ssnmf[0], ssnmf[1]) * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b)))))

      print("GENERATE: " + r_pathes[i])
      librosa.output.write_wav(r_pathes[i], result, sr_b)

  def __nmf(self, Y, R=3, n_iter=50, init_H=[], init_U=[], verbose=False):
    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0]
    N = Y.shape[1]

    # initialization
    if len(init_U):
      U = init_U
      R = init_U.shape[0]
    else:
      U = np.random.rand(R,N)

    if len(init_U):
      H = init_H
      R = init_H.shape[1]
    else:
      H = np.random.rand(M,R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)

    # computation of Lambda (estimate of Y)
    Lambda = np.dot(H, U)

    # iterative computation
    for i in range(n_iter):
      #compute euclid_divergence
      cost[i] = self.__euclid_divergence(Y, Lambda)

      # update H and U
      H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)
      U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)

      # recomputation of Lambda
      Lambda = np.dot(H, U)

    return [H, U, cost]

  def __ssnmf(self, Y, R=3, n_iter=50, F=[], init_G=[], init_H=[], init_U=[], verbose=False):
    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0]
    N = Y.shape[1]
    X = F.shape[1]

    # initialization
    if len(init_G):
      G = init_G
      X = init_G.shape[1]
    else:
      G = np.random.rand(X, N)

    if len(init_U):
      U = init_U
      R = init_U.shape[0]
    else:
      U = np.random.rand(R, N)

    if len(init_H):
      H = init_H
      R = init_H.shape[1]
    else:
      H = np.random.rand(M, R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)

    # computation of Lambda(estimate of Y)
    Lambda = np.dot(F, G) + np.dot(H, U)

    # iterative computation
    for it in range(n_iter):
      # compute euclid divergence
      cost[it] = self.__euclid_divergence(Y, Lambda + eps)

      # update H, U and G
      H *= (np.dot(Y, U.T) + eps) / (np.dot(np.dot(H, U) + np.dot(F, G), U.T) + eps)
      U *= (np.dot(H.T, Y) + eps) / (np.dot(H.T, np.dot(H, U) + np.dot(F, G)) + eps)
      G *= (np.dot(F.T, Y) + eps)[np.arange(G.shape[0])] / (np.dot(F.T, np.dot(H, U) + np.dot(F, G)) + eps)

      # recomputation of Lambda(estimate of V)
      Lambda = np.dot(H, U) + np.dot(F, G)

    return [F, G, H, U, cost]

  def __euclid_divergence(self, Y, Yh):
    return (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum() / 2

  def __export_wav(self,path,samples,width,frame_rate,channels):

    sound = AudioSegment(
      samples.astype("int16").tobytes(),
      sample_width=width,
      frame_rate=frame_rate,
      channels=channels,
    )

    sound.export(path,format="wav")

def isfloat(parameter):
  if not parameter.isdecimal():
    try:
      float(parameter)
      return True
    except ValueError:
      return False
  else:
    return False

# if __name__ == "__main__":
#   dirs = glob.glob("../data/dataset/*")
#   SS = SoundSeparation()

#   for dir in dirs:
#     # if os.path.exists(dir + "/LR.wav"):
#     #   SS.double_to_single(
#     #     dir + "/LR.wav",
#     #     dir + "/0.wav",
#     #     dir + "/1.wav"
#     #   )

#     with open(dir + "/sections.csv", "r") as f:
#       target_sounds = glob.glob(dir + "/[0~1].wav")
#       reader = csv.reader(f)
#       index = 0

#       for row in reader:
#         if not isfloat(row[0]):
#           continue

#         start = int(float(row[0]) * 1000)
#         end = int(float(row[1]) * 1000)
        
#         for i, target in enumerate(target_sounds):
#           sound = AudioSegment.from_file(target, format="wav")
#           filepath = dir + "/" + str(i) + "_" + str(index) + ".wav"
#           print(filepath)

#           # if not os.path.exists(filepath):
#           temp = sound[start:end]
#           temp.export(filepath, format="wav")

#         index += 1

#       fileid = os.path.basename(dir).split('_')
#       state_dir = "../data/state_list/"
      
#       for x, id in enumerate(fileid):
#         sections = glob.glob(dir + "/" + str(x) + "_?.wav")
#         concat_dataset = []
        
#         for i, section in enumerate(sections):
#           target_states = state_dir + str(int(id)) + "_" + str(i+1) + ".csv"
#           print("TARGET: " + target_states)
#           print(os.path.exists(section))
#           sound = AudioSegment.from_file(section, format="wav")
#           if not os.path.exists(target_states):
#             continue

#           with open(target_states, "r") as f:
#             reader = csv.reader(f)
#             start, end = 0.0, 0.0
#             for row in reader:
#               if len(row) <= 0:
#                 continue

#               start = end
#               end += float(row[1])

#               if int(row[0]) is 1:
#                 concat_dataset.append(sound[start*1000:end*1000])
        
#         if len(concat_dataset) > 0:
#           result = concat_dataset[0]
#           for data in concat_dataset[1:]:
#             result = result + data

#           result_path = "../data/ss_train/" + id + ".wav"
#           print("RESULT: " + result_path)
#           result.export(result_path, format="wav")

if __name__ == "__main__":
  # np.random.seed(1)
  # comps = np.array(((1,0), (0,1), (1,1)))
  # activs = np.array(((0,0,1,0,1,5,0,7,9,6,5,0), (2,1,0,1,1,2,1,0,0,0,6,0)))
  # Y = np.dot(comps, activs)

  # print('original data\n---------------')
  # print('components:\n', comps)
  # print('activations:\n', activs)
  # print('Y:\n', Y)

  # SS = SoundSeparation()
  # computed = SS.nmf(Y, R=2)

  # print('\ndecomposed\n---------------')
  # print('H:\n', computed[0])
  # print('U:\n', computed[1])
  # print('HU:\n', np.dot(computed[0], computed[1]))
  # print('cost:\n', computed[2]) 

  # np.random.seed(1)
  # comps = np.array(((1,0), (0,1), (1,1)))
  # activs = np.array(((0,0,1,0,1,5,0,7,9,6,5,0), (2,1,0,1,1,2,1,0,0,0,6,0)))
  # Y = np.dot(comps, activs)

  # print('original data\n---------------')
  # print('components:\n', comps)
  # print('activations:\n', activs)
  # print('Y:\n', Y)

  # SS = SoundSeparation()
  # computed = SS.ssnmf(Y, R=2, F=np.array(((1, 0, 1),)).T)

  # print('\ndecomposed\n---------------')
  # print('F:\n', computed[0])
  # print('G:\n', computed[1])
  # print('H:\n', computed[2])
  # print('U:\n', computed[3])
  # print('FG + HU:\n', np.dot(computed[0], computed[1]) + np.dot(computed[2], computed[3]))
  # print('cost:\n', computed[4])

  SS = SoundSeparation()
  
  # dir pathes
  train_dataset = "../data/ss_train/"
  predict_pathes = glob.glob("../data/dataset/*")

  process_count = 39 * 4
  R = [list(range(300,900,100)), list(range(100,300,50))]

  executer = concurrent.futures.ProcessPoolExecutor()
  used_thread_count = 0
  MAX_THREAD_COUNT = multiprocessing.cpu_count()
  features = []

  for R_t in R[0]:
    for R in R[1]:

      for predict in predict_pathes:
        fileid = os.path.basename(predict).split("_")

        timestamp = '{0:%y%m%d%H%M}'.format(datetime.datetime.now())
        result_path = "../data/ss_result/" + timestamp + "/"
        if not os.path.exists(result_path):
          os.mkdir(result_path)

        for i, id in enumerate(fileid):
          predict_pathes, result_pathes = [], []
          for j in range(4):
            predict_pathes.append(predict + "/" + str(i) + "_" + str(j) + ".wav")
            result_pathes.append(result_path + id + "_" + str(j) + ".wav")

          features.append(executer.submit(
            SS.separate,
            R_t,
            750,
            R,
            300,
            predict_pathes, 
            train_dataset + id + ".wav", 
            result_pathes
          ))
          used_thread_count += 1

          if used_thread_count is MAX_THREAD_COUNT:
            while len(features) >= MAX_THREAD_COUNT:
              for f in concurrent.futures.as_completed(features):
                f.result()
                features.remove(f)
                used_thread_count -= 1
  