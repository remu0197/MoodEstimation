import numpy as np 
from scipy.io.wavfile import read, write

def main() :
    wavfile = "../data/1905221218/2_l.wav"
    filename = "../data/1905221218/test.wav"

    fs, x = read(wavfile)
    N = len(x)
    dt = 0.01
    t = np.arange(0, N*dt, dt)
    freq = np.linspace(0, 1.0/dt, N)
    fc = 20

    F = np.fft.fft(x)
    F = F / (N / 2)
    F[0] = F[0] / 2

    F2 = F.copy()
    F2[(freq < fc)] = 0
    F2[(freq > 1/(dt*2))] = 0
    f2 = np.fft.ifft(F2)
    f2 = np.real(f2*N)

    write(filename, fs, f2)

if __name__ == "__main__":
    main()    