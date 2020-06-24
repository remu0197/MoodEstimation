import numpy as np
import librosa
import scipy.io.wavfile as wav
from pydub import AudioSegment
import glob, os, csv, datetime
import concurrent.futures
import multiprocessing, copy
import sys, wave

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

  def separate(self,R_t,n_iter_t,R,n_iter,predict_pathes, train_pathes, r_pathes):
    # for train_path in train_pathes:
    #   if not os.path.exists(train_path):
    #     return

    T = 10

    base = []
    for train_path in train_pathes:
      print("   TRAINING: " + train_path)
      y_train, _ = librosa.load(train_path)
      s_train = librosa.stft(y_train)
      nmf = self.__nmf(np.abs(s_train), R=R_t, n_iter=n_iter_t)
      print(len(nmf[0]))
      base.append(nmf[0])

    # print("   TRAINING: " + train_pathes)
    # y_train, _ = librosa.load(train_pathes)
    # s_train = librosa.stft(y_train)
    # nmf = self.__nmf(np.abs(s_train), R=R_t, n_iter=n_iter_t)
    # base = nmf[0]

    for i, predict in enumerate(predict_pathes):
      if not os.path.exists(predict):
        continue

      print("   PRIDICT: " + predict)
      y_b, sr_b = librosa.load(predict)
      s_b = librosa.stft(y_b)

      # SSNMF
      # ssnmf = self.__ssnmf(np.abs(s_b), R=R, F=base, n_iter=n_iter)    
      # # result = librosa.istft((np.dot(ssnmf[0], ssnmf[1]) * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b)))))

      # 2SNMF
      ssnmf = self.__2snmf(np.abs(s_b), R=R, F=base, n_iter=n_iter)
      result = librosa.istft((np.dot(ssnmf[0], ssnmf[1]) * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b)))))

      # CNMF
      # cnmf = self.__cnmf(np.abs(s_b), R=R, T=10, n_iter=n_iter)
      # result = librosa.istft(cnmf[0] * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b))))

      # SSCNMF
      # sscnmf = self.__sscnmf(np.abs(s_b),R=R,T=T,Ft=base,n_iter=n_iter)
      # result = librosa.istft(sscnmf[0] * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b))))

      # 2SSCNMF
      # sscnmf = self.__2sscnmf(np.abs(s_b),R=R,T=T,Ft=base,n_iter=n_iter)
      # result = librosa.istft(sscnmf[0] * np.cos(np.angle(s_b) + 1j * np.sin(np.angle(s_b))))

      print("   GENERATE: " + r_pathes[i])
      librosa.output.write_wav(r_pathes[i], result, sr_b)

  def __max_db(self, path):
    max_db = 0
    N = 1024

    wave_file = wave.open(path, "rb")
    x = wave_file.readframes(wave_file.getnframes())
    x = np.frombuffer(x, dtype="int16")

    pad = np.zeros(N//2)
    pad_data = np.concatenate([pad, x, pad])
    rms = np.array([np.sqrt((1/N) * (np.sum(pad_data[i:i+N]))**2) for i in range(len(x))])
    dbs = 20 * np.log10(rms)

    for db in dbs:
      if max_db < db:
        max_db = db

    return max_db

  def align_db(self, base_sound_path, edit_sound_path, output_sound_path="", sound_format="wav"):
    base_db, edit_db = self.__max_db(base_sound_path), self.__max_db(edit_sound_path)
    edit_sound = AudioSegment.from_file(edit_sound_path, format=sound_format)

    edit_sound = edit_sound + (base_db - edit_db)

    if output_sound_path == "":
      edit_sound.export(edit_sound_path, format=sound_format)
    else:
      edit_sound.export(output_sound_path, format=sound_format)

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
      H_prev = copy.deepcopy(H)
      U_prev = copy.deepcopy(U)

      # update H, U and G
      H *= (np.dot(Y, U_prev.T) + eps) / (np.dot(Lambda, U_prev.T) + eps)
      U *= (np.dot(H_prev.T, Y) + eps) / (np.dot(H_prev.T, Lambda) + eps)
      G *= (np.dot(F.T, Y) + eps) / (np.dot(F.T, Lambda) + eps)

      # recomputation of Lambda(estimate of V)
      Lambda = np.dot(F, G) + np.dot(H, U)

    return [F, G, H, U, cost]

  def __2snmf(self, Y, F, R=3, n_iter=50, init_G=[], init_H=[], init_U=[], verbose=False):
    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0]
    N = Y.shape[1]
    # print(F[0].shape)
    X = F[0].shape[1]

    # initialization
    if len(init_G):
      G = init_G
      # X = init_G.shape[1]
    else:
      G = np.random.rand(X, N)

    G_2 = np.random.rand(X, N)

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
    Lambda = np.dot(F[0], G) + np.dot(F[1], G_2) + np.dot(H, U)
    print(F[0].shape)

    # iterative computation
    for it in range(n_iter):
      # compute euclid divergence
      cost[it] = self.__euclid_divergence(Y, Lambda + eps)
      H_prev = copy.deepcopy(H)
      U_prev = copy.deepcopy(U)

      # update H, U and G
      H *= (np.dot(Y, U_prev.T) + eps) / (np.dot(Lambda, U_prev.T) + eps)
      U *= (np.dot(H_prev.T, Y) + eps) / (np.dot(H_prev.T, Lambda) + eps)
      G *= (np.dot(F[0].T, Y) + eps) / (np.dot(F[0].T, Lambda) + eps)
      G_2 *= (np.dot(F[1].T, Y) + eps) / (np.dot(F[1].T, Lambda) + eps)

      # recomputation of Lambda(estimate of V)
      Lambda = np.dot(F[0], G) + np.dot(F[1], G_2) + np.dot(H, U)

    return [F[0], G]
  
  def __cnmf(self, Y, R, T, n_iter):
    # array to save the value of the costs
    cost = np.zeros(n_iter)
    [M, N] = Y.shape

    Ht = [np.random.rand(M, R) * 2 - 1 for _ in range(T)]
    U = np.random.rand(R, N) * 2 - 1
    Ut = self.__get_t_array(array=U, T=T, is_right=True)
    Yt = self.__get_t_array(array=Y, T=T, is_right=False)

    Yh = np.dot(Ht[0],Ut[0])
    for t in range(1,T):
      Yh += np.dot(Ht[t],Ut[t])
    Yht = self.__get_t_array(array=Yh, T=T, is_right=False)

    for it in range(n_iter):
      cost[it] = self.__euclid_divergence(Y, Yh)
      U_temp = np.zeros(U.shape)

      for t in range(T):
        Ht[t] *= np.dot(Y, Ut[t].T) / np.dot(Yh, Ut[t].T)
        A = np.dot(Ht[t].T, Yt[t])
        B = np.dot(Ht[t].T, Yht[t])
        U_temp += U * np.divide(A,B,out=np.zeros_like(A),where=B!=0)

      U = U_temp / T
      U_max, U_min = U.max(), U.min()
      for row in range(len(U)):
        for index in range(len(U[row])):
          U[row][index] = (U[row][index] - U_min) / (U_max - U_min)
      Ut = self.__get_t_array(array=U, T=T, is_right=True)

      Yh = np.dot(Ht[0],Ut[0])
      for t in range(1,T):
        Yh += np.dot(Ht[t],Ut[t])
      Yht = self.__get_t_array(array=Yh, T=T, is_right=False)  

    return [Ht, Ut, cost]

  def __sscnmf(self, Y, R, T, Ft, n_iter):
    cost = np.zeros(n_iter)
    [M,N] = Y.shape

    G = np.random.rand(R,N) * 2 - 1
    Gt = self.__get_t_array(array=G, T=T, is_right=True)
    Ht = [np.random.rand(M, R) * 2 - 1 for _ in range(T)]
    U = np.random.rand(R,N) * 2 - 1
    Ut = self.__get_t_array(array=U, T=T, is_right=True)
    Yt = self.__get_t_array(array=Y, T=T, is_right=False)

    Yh = np.dot(Ft[0], Gt[0]) + np.dot(Ht[0], Ut[0])
    for t in range(1, T):
      Yh += np.dot(Ft[t], Gt[t]) + np.dot(Ht[t], Ut[t])
    Yht = self.__get_t_array(array=Yh, T=T, is_right=False)

    for it in range(n_iter):
      cost[it] = self.__euclid_divergence(Y, Yh)
      U_temp = np.zeros(U.shape)
      G_temp = np.zeros(G.shape)
      Ht_prev = copy.deepcopy(Ht)
      Ut_prev = copy.deepcopy(Ut)

      for t in range(T):
        A = np.dot(Yt[0], Ut_prev[t].T)
        B = np.dot(Yht[0], Ut_prev[t].T)
        Ht[t] *= np.divide(A,B,out=np.zeros_like(A),where=B!=0)
        A = np.dot(Ht_prev[t].T, Yt[t])
        B = np.dot(Ht_prev[t].T, Yht[t])
        U_temp += U * np.divide(A,B,out=np.zeros_like(A),where=B!=0)
        C = np.dot(Ft[t].T, Yt[t])
        D = np.dot(Ft[t].T, Yht[t])
        G_temp += G * np.divide(C,D,out=np.zeros_like(C),where=D!=0)

      U = U_temp / T
      U_max, U_min = U.max(), U.min()
      for row in range(len(U)):
        for index in range(len(U[row])):
          U[row][index] = (U[row][index] - U_min) / (U_max - U_min)
      Ut = self.__get_t_array(array=U,T=T,is_right=True)
      
      G = G_temp / T
      G_max, G_min = G.max(), G.min()
      for row in range(len(G)):
        for index in range(len(G[row])):
          G[row][index] = (G[row][index] - G_min) / (G_max - G_min)
      Gt = self.__get_t_array(array=G,T=T,is_right=True)

      Yh = np.dot(Ft[0], Gt[0]) + np.dot(Ht[0], Ut[0])
      for t in range(1, T):
        Yh += np.dot(Ft[t], Gt[t]) + np.dot(Ht[t], Ut[t])
      Yht = self.__get_t_array(array=Yh, T=T, is_right=False)  

    result = np.dot(Ft[0], Gt[0])
    for t in range(1,T):
      result += np.dot(Ft[t],Gt[t])

    return [result, cost]

  def __2sscnmf(self, Y, R, T, Ft, n_iter):
    cost = np.zeros(n_iter)
    [M,N] = Y.shape
    R_1, R_2 = R, R

    G_1 = np.random.rand(R_1,N) * 2 - 1
    Gt_1 = self.__get_t_array(array=G_1, T=T, is_right=True)
    G_2 = np.random.rand(R_2,N) * 2 - 1
    Gt_2 = self.__get_t_array(array=G_2, T=T, is_right=True)
    Ht = [np.random.rand(M, R) * 2 - 1 for _ in range(T)]
    U = np.random.rand(R,N) * 2 - 1
    Ut = self.__get_t_array(array=U, T=T, is_right=True)
    Yt = self.__get_t_array(array=Y, T=T, is_right=False)

    Yh = np.dot(Ft[0][0], Gt_1[0]) + np.dot(Ft[1][0], Gt_2[0]) + np.dot(Ht[0], Ut[0])
    for t in range(1, T):
      Yh += np.dot(Ft[0][t], Gt_1[t]) + np.dot(Ft[1][t], Gt_2[t]) + np.dot(Ht[t], Ut[t])
    Yht = self.__get_t_array(array=Yh, T=T, is_right=False)

    for it in range(n_iter):
      cost[it] = self.__euclid_divergence(Y, Yh)
      U_temp = np.zeros(U.shape)
      G_1_temp = np.zeros(G_1.shape)
      G_2_temp = np.zeros(G_2.shape)
      Ht_prev = copy.deepcopy(Ht)
      Ut_prev = copy.deepcopy(Ut)

      for t in range(T):
        A = np.dot(Yt[0], Ut_prev[t].T)
        B = np.dot(Yht[0], Ut_prev[t].T)
        Ht[t] *= np.divide(A,B,out=np.zeros_like(A),where=B!=0)
        A = np.dot(Ht_prev[t].T, Yt[t])
        B = np.dot(Ht_prev[t].T, Yht[t])
        U_temp += U * np.divide(A,B,out=np.zeros_like(A),where=B!=0)
        A = np.dot(Ft[0][t].T, Yt[t])
        B = np.dot(Ft[0][t].T, Yht[t])
        G_1_temp += G_1 * np.divide(A,B,out=np.zeros_like(A),where=B!=0)
        A = np.dot(Ft[1][t].T, Yt[t])
        B = np.dot(Ft[1][t].T, Yht[t])
        G_2_temp += G_2 * np.divide(A,B,out=np.zeros_like(A),where=B!=0)

      U = U_temp / T
      U_max, U_min = U.max(), U.min()
      for row in range(len(U)):
        for index in range(len(U[row])):
          U[row][index] = (U[row][index] - U_min) / (U_max - U_min)
      Ut = self.__get_t_array(array=U,T=T,is_right=True)
      
      G_1 = G_1_temp / T
      G_1_max, G_1_min = G_1.max(), G_1.min()
      for row in range(len(G_1)):
        for index in range(len(G_1[row])):
          G_1[row][index] = (G_1[row][index] - G_1_min) / (G_1_max - G_1_min)
      Gt_1 = self.__get_t_array(array=G_1,T=T,is_right=True)

      G_2 = G_2_temp / T
      G_2_max, G_2_min = G_2.max(), G_2.min()
      for row in range(len(G_2)):
        for index in range(len(G_2[row])):
          G_2[row][index] = (G_2[row][index] - G_2_min) / (G_2_max - G_2_min)
      Gt_2 = self.__get_t_array(array=G_2,T=T,is_right=True)

      Yh = np.dot(Ft[0][0], Gt_1[0]) + np.dot(Ft[1][0], Gt_2[0]) + np.dot(Ht[0], Ut[0])
      for t in range(1, T):
        Yh += np.dot(Ft[0][t], Gt_1[t]) + np.dot(Ft[1][t], Gt_2[t]) + np.dot(Ht[t], Ut[t])
      Yht = self.__get_t_array(array=Yh, T=T, is_right=False)

    result = np.dot(Ft[0][0], Gt_1[0])
    for t in range(1,T):
      result += np.dot(Ft[0][t], Gt_1[t])

    return [result, cost]

  def __get_t_array(self, array, T, is_right=True):
    M = array.shape[0]
    N = array.shape[1]
    zeros = [[0] for _ in range(M)]
    t_array = [copy.deepcopy(array)]
    temp = copy.deepcopy(array)

    if is_right:
      for _ in range(T):
        temp = np.concatenate([zeros, temp], 1)
        temp = np.delete(temp,N,1)
        t_array.append(copy.deepcopy(temp))
    else:
      for _ in range(T):
        temp = np.concatenate([temp, zeros], 1)
        temp = np.delete(temp,0,1)
        t_array.append(copy.deepcopy(temp))

    return t_array

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
  is_one_roop = False
  
  # dir pathes
  train_dataset = "../data/ss_train/"
  predict_dirs = glob.glob("../data/dataset/*")

  process_count = 39 * 4 * 5 * 6
  completed_count = 0
  R = [[800], [800]]
  # R = [list(range(50,150,50)), list(range(50,150,50))]
  # R = [list(range(300,900,100)), list(range(100,300,50))]

  used_thread_count = 0
  MAX_THREAD_COUNT = int(multiprocessing.cpu_count() / 2)
  executer = concurrent.futures.ProcessPoolExecutor(MAX_THREAD_COUNT)
  features = []

  for R_t in R[0]:
    for R_p in R[1]:

      timestamp = '2snmf'#'{0:%y%m%d%H%M}'.format(datetime.datetime.now())
      result_path = "../data/ss_result/" + timestamp +"_" + str(R_t) + "_" + str(R_p) + "/"
      if not os.path.exists(result_path):
        os.mkdir(result_path)

      for predict in predict_dirs:
        fileid = os.path.basename(predict).split("_")

        for i, id in enumerate(fileid):
          predict_pathes, result_pathes = [], []

          if not os.path.exists(train_dataset + id + "_1.wav"):
            continue

          for j in range(4):
            predict_pathes.append(predict + "/" + str(i) + "_" + str(j) + ".wav")
            result_pathes.append(result_path + id + "_" + str(j) + ".wav")

          SS.separate(
            R_t,
            3000,
            R_p,
            3000,
            predict_pathes,
            [train_dataset + id + "_1.wav", train_dataset + id + "_2.wav"],
            result_pathes
          )

          break
        break
      break
    break
 
          # features.append(executer.submit(
          #   SS.separate,
          #   R_t,
          #   10000,
          #   R_p,
          #   10000,
          #   predict_pathes, 
          #   [train_dataset + id + "_1.wav", train_dataset + id + "_2.wav"],
          #   # train_dataset + id + "_1.wav",
          #   result_pathes
          # ))
          # used_thread_count += 1

          # if used_thread_count is MAX_THREAD_COUNT:
          #   while len(features) >= MAX_THREAD_COUNT:
          #     for f in concurrent.futures.as_completed(features):
          #       f.result()
          #       features.remove(f)
          #       used_thread_count -= 1
          #       completed_count += 1
          #       print("COMPLETED: " + str(completed_count) + " / " + str(process_count))

              
