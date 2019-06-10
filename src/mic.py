import pyaudio
import wave
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(16e3)

'''
function records audio from the default microphone for length 't' seconds and sample rate 'sr', and deposits a .wav file
named 'file_name' in the directory specified by 'path'
'''


class VoiceRecorder:
    def __init__(self, file_name='test_record.wav', path='../data/raw', max_time_sec=20):
        self.frames = []
        self.max_time = max_time_sec
        self.output_name = os.path.join(path, file_name)
        self.path = path
        self.sr = RATE
        self.p = pyaudio.PyAudio()
        self.wf = None
        self.open()

    def open(self):
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)


    def start_record(self):
        for i in range(0, int(RATE / CHUNK) * self.max_time):
            data = self.stream.read(CHUNK)
            self.frames.append(data)
        print(f"finished recording for {self.max_time}")

    def stop_record(self):
        print(f"Prematurely finished recording")
        self.stream.stop_stream()

    def close_record(self):
        self.stream.close()
        self.p.terminate()

    def recording(self):
        return self.stream.is_active()

    def save_wave(self):
        self.wf = wave.open(self.output_name, 'wb')
        self.wf.setnchannels(CHANNELS)
        self.wf.setframerate(RATE)
        self.wf.setsampwidth(self.p.get_sample_size(FORMAT))
        self.wf.writeframes(b''.join(self.frames))
        self.wf.close()


def bokeh_test_record():
    from bokeh.models import Div, ColumnDataSource, LinearColorMapper
    from bokeh.plotting import figure, curdoc
    from bokeh.models.widgets import Toggle
    from src.player import AudioPlayer
    from bokeh.layouts import column, row, grid
    import librosa
    import os
    import numpy as np
    import pandas as pd
    from PIL import Image
    width = 800
    height = 200
    vr = VoiceRecorder()
    src = ColumnDataSource({'time':np.arange(0,100,1),'amplitude':np.random.randn(1,100)})
    p = figure(plot_width=width, plot_height=height, title="Audio Waveform",
               toolbar_location='above', tools=[], output_backend="webgl")
    p.line('time','amplitude',source = src)
    toggle_record = Toggle(label='Record', active=False)

    def callback_record(arg):
        p.title.text = f'{toggle.active}'
        if toggle_record.active:
            toggle_record.label = 'Recording..'
            toggle_record.button_type = 'warning'
            vr.start_record()
        else:
            toggle_record.label = 'Record via mic'
            toggle_record.button_type = 'default'
            if vr.recording():
                vr.stop_record()
                toggle_record.label = 'Done recording'



    toggle_record.on_click(callback_record)
    curdoc().add_root(grid(column(toggle_record, p)))


bokeh_test_record()
#
# def record(file_name, path, t=10, sr=16000):
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = sr
#     p = pyaudio.PyAudio()
#
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)
#
#     frames = []
#     print("starting recording " + file_name + ".")
#     for i in range(0, int(RATE / CHUNK )*t):
#         data = stream.read(CHUNK)
#         frames.append(data)
#     print("finished recording")
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     os.makedirs(path,exist_ok=True)
#     fname = os.path.join(path,file_name)
#     wf = wave.open(fname, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()
#
# record(file_name='test_record.wav',path = '../data/raw')
