from bokeh.plotting import figure, show,curdoc
from bokeh.events import Tap
from bokeh.models.callbacks import CustomJS
from bokeh.models import ColorBar, LinearColorMapper, Div, HoverTool, ColumnDataSource, TapTool
import sys
sys.path.insert(0,'D:\\Repos\\diurisation-pet-proj')
from bokeh.io import save
from bokeh.client import push_session
from bokeh.models.widgets import Toggle
from src.player import AudioPlayer
from bokeh.layouts import row, column
from bokeh.palettes import RdBu11, OrRd9,Blues9
from bokeh.transform import linear_cmap
import librosa
import os
import numpy as np
import pandas as pd


fname = 'D:\\Repos\\diurisation-pet-proj\\data\\raw\\carell.wav'
filename_base =os.path.splitext(os.path.basename(fname))[0]
x, sr = librosa.load(fname)
df = pd.DataFrame({"time":np.arange(x.shape[0])/sr,"amplitude":x})

df_small = df.sample(frac = 0.01).sort_values('time')
max_time = df['time'].max()
dt = 100 # in ms
print(df_small.shape)
s1 = ColumnDataSource(df_small)
s2 = ColumnDataSource({'x':[df['time'].min(),df['time'].min()],'y':[-0.5,0.5]})
p = figure(plot_width=700, plot_height=150, title="Audio Waveform",
           toolbar_location='above', tools=[])
p.line(x='time',y = 'amplitude',source = s1)
l = p.line(x='x',y='y',source = s2,line_width=2,color='black')


d = Div()
play = False
toggle = Toggle(label='Play',active=False)
# noinspection PyUnresolvedReferences
def callback_tap(s2=s2):
    xpos = cb_obj.x
    data = s2.data
    print(xpos)
    data['x']=[xpos,xpos]
    s2.data =data
    s2.change.emit()

def callback_play(arg):
    #print(old)
  #  global play
   # play = not play
    p.title.text =f'{toggle.active}'
    if toggle.active:
        toggle.label = 'Pause'
        toggle.button_type='success'
    else:
        toggle.label = 'Play'
        toggle.button_type = 'default'

def update():
    #global play
    #if play:
    if toggle.active:
        if s2.data['x'][0]+dt/1000 > max_time:
            #play = False
            toggle.active = False
            s2.data['x'] = [0*i for i in s2.data['x']]
            d.text = f' Exceed time: {play}'
        else:
            d.text = f' Else: {play}'
            s2.data['x'] = [i + dt/1000 for i in s2.data['x']]


curdoc().add_periodic_callback(update, dt)
p.js_on_event('tap',CustomJS.from_py_func(callback_tap))
toggle.on_click(callback_play)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(toggle, p,d))
