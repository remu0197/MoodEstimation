def test(*args) :
    for s in args :
        print(s)

test('a', 'b', 'c')

import numpy as np 
import scipy.io.wavfile as wf

class VAD :
    def __init__(self, data) :
        self.__epsilon = 1e-5
        self.__x = np.matrix(data)
        self.__rates = []

    def ica(self) :
        self.fit()
        z = self.whiten()
        y = self.analyze(z)

        return y

    def fit(self) :
        self.__x -= self.__x.mean(axis=1)

    def whiten(self) :
        sigma = np.cov(self.__x, rowvar=True, bias=True)
        D, E = np.linalg.eigh(sigma)
        E = np.asmatrix(E)
        Dh = np.diag(np.array(D)**(-1/2))
        V = E * Dh * E.T
        z = V * self.__x
        return z

    def normalize(self, x) :
        if x.sum() < 0 :
            x *= -1

        return x / np.linalg.norm(x)

    def analyze(self, z) :
        c, _ = self.__x.shape
        W = np.empty((0, c))
        for _ in range(c) :
            vec_w = np.random.rand(c, 1)
            vec_w = self.normalize(vec_w)
            while True :
                vec_w_prev = vec_w
                vec_w = np.asmatrix((np.asarray(z) + np.asarray(vec_w.T * z) ** 3).mean(axis=1)).T - 3 * vec_w
                vec_w = self.normalize(np.linalg.qr(np.asmatrix(np.concatenate((W, vec_w.T), axis=0)).T)[0].T[-1].T)

                if np.linalg.norm(vec_w - vec_w_prev) < self.__epsilon :
                    W = np.concatenate((W, vec_w.T), axis=0)
                    break

        y = W * z
        return y

def main() :
    rate1, data1 = wf.read('../data/test/0_l.wav')
    rate2, data2 = wf.read('../data/test/0_r.wav')

    data = [data1.astype(float), data2.astype(float)]
    y = VAD(data).ica()
    y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.array(y)]

    wf.write('../data/test/result_l.wav', rate1, y[0])
    wf.write('../data/test/result_r.wav', rate2, y[1])

if __name__ == '__main__' :
    main()
