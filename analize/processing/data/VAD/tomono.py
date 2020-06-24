import os
import wave
import audioop
import pyaudio

def wav_write(buf, path) :
    audio = pyaudio.PyAudio()

    wave_file = wave.open(path, "w")
    wave_file.setnchannels(1)
    wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(48000)
    wave_file.writeframes(b''.join(buf))
    wave_file.close()

dirlist = os.listdir('./data/')

for d in dirlist :
    filelist = os.listdir('./data/' + d)
    
    for file in filelist :
        wav = wave.open('./data/' + d + '/' + file, 'r')
        buf = wav.readframes(-1)
        buf_r = audioop.tomono(buf, 2, 1, 0)
        buf_l = audioop.tomono(buf, 2, 0, 1)
        wav.close()

        s = file.strip('.wav')
        wav_write(buf_r, './data/' + d + '/' + s + '_r.wav')
        wav_write(buf_l, './data/' + d + '/' + s + '_l.wav')
