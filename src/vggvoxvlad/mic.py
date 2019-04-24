import pyaudio
import wave
import os

'''
function records audio from the default microphone for length 't' seconds and sample rate 'sr', and deposits a .wav file
in the directory specified by 'path'
'''


def record(file_name, path, t=10, sr=16000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sr
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    print("starting recording " + file_name + ".")
    for i in range(0, int(RATE / CHUNK) * t):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    fpath = os.path.join(path, (file_name + '.wav'))
    print(fpath)
    wf = wave.open(fpath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

