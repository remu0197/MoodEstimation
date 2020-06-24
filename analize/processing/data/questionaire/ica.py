import sys
from enum import Enum
import numpy as np
import scipy.io.wavfile as wav 

class ICA :
    def __init__(self, x):
        self.__sources = None
        self.__s = []
        self.__fs = []
        self.__epsilon = 1e-5
        self.__is_separated = False
        return

    def read_source(self, path) :
        fs, s = wav.read(path)
        self.__fs.append(fs)
        self.__s.append(s)

    def read_sources(self, pathes) :
        if len(pathes) == 0 :
            print('Can not read sources: value is not array')
            return False

        for path in pathes :
            self.read_source(path)

    def separate(self) :
        if len(self.__s) < 0 :
            print('Not enough sources')
            return False

        data = []
        for s in self.__s :
            s = s.astype(float)
            data.append(s)
        
        self.__sources = np.matrix(data)
        self.__fit()
        z = self.__whiten()
        y = self.__analyze(z)

        return y

    def __fit(self) :
        self.__sources -= self.__sources.mean(axis=1)

    def __whiten(self) :
        sigma = np.cov(self.__sources, rowvar=True, bias=True)
        D, E = np.linalg.eigh(sigma)
        E = np.asmatrix(E)
        Dh = np.diag(np.array(D) ** (-1/2))
        V = E * Dh * E.T
        z = V * self.__sources
        return z

    def __normalize(self, x):
        if x.sum() < 0:
            x *= -1
        return x / np.linalg.norm(x)

    def __analyze(self, z) :
        c, _ = self.__sources.shape
        W = np.empty((0, c))
        for _ in range(c): #観測数分だけアルゴリズムを実行する
            vec_w = np.random.rand(c, 1)
            vec_w = self.__normalize(vec_w)
            while True:
                vec_w_prev = vec_w
                vec_w = np.asmatrix((np.asarray(z) * np.asarray(vec_w.T * z) ** 3).mean(axis=1)).T - 3 * vec_w
                vec_w = self.__normalize(np.linalg.qr(np.asmatrix(np.concatenate((W, vec_w.T), axis=0)).T)[0].T[-1].T) #直交化法と正規化
                if np.linalg.norm(vec_w - vec_w_prev) < self.__epsilon: #収束判定
                    W = np.concatenate((W, vec_w.T), axis=0)
                    break
        y = W * z
        return y

