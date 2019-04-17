import sys

from bokeh.models import Div, ColumnDataSource,CategoricalColorMapper,Jitter
from bokeh.palettes import d3
from bokeh.plotting import figure, curdoc

sys.path.insert(0,'D:\\Repos\\diurisation-pet-proj')
from bokeh.models.widgets import Toggle
from src.player import AudioPlayer
from bokeh.layouts import column
import librosa
import os
import numpy as np
import pandas as pd
from src.sad_diarisation_bk import load_rttm,label_waveform_by_speaker
# def load_rttm(rttm_file_name):
#     speaker_track = pd.read_csv(fname_rttm, header=None, delim_whitespace=True,
#                                 names=['x1', 'filename', 'fileid', 'start', 'duration', 'skip1', 'skip2', 'speaker_id',
#                                        'skip3'])
#     speaker_track['end'] = speaker_track['start'] + speaker_track['duration']
#     speaker_track['id'] = speaker_track['speaker_id'].apply(lambda x: int(x.split('speaker')[1]))
#
#     bins_speaker = pd.DataFrame({'name': speaker_track['speaker_id'].values},
#                                 index=pd.IntervalIndex.from_arrays(left=speaker_track['start'],
#                                                                    right=speaker_track['end']))
#     speaker_track.drop(['skip1', 'skip2', 'skip3'], axis=1, inplace=True)
#     return speaker_track
#     #cat_type = CategoricalDtype(categories=speaker_track['speaker_id'].unique().tolist())

width=1000
height=250


fname = '../data/raw/carell.wav'
fname_rttm = '../data/processed/carell/carell.rttm'
rttm_df = load_rttm(fname_rttm,min_duration=0.2)

print(rttm_df.head())
filename_base =os.path.splitext(os.path.basename(fname))[0]
x, sr = librosa.load(fname)
ap = AudioPlayer(fname)
fr = ap.wf.getframerate()
df = pd.DataFrame({"time":np.arange(x.shape[0])/(sr),"amplitude":x})
max_time = df['time'].max()
df_small = df.sample(frac = 0.01).sort_values('time')
df_small['amplitude'] = (df_small['amplitude'] - df_small['amplitude'].min()) / (df_small['amplitude'].max() - df_small['amplitude'].min()) - 0.5
df_small = label_waveform_by_speaker(df_small,rttm_df)
dt = 50 # in ms
print(df_small.sample(frac=1).head())
print(df_small.shape)


palette = d3['Category10'][len(df_small['ID'].unique())]
color_map = CategoricalColorMapper(factors=df_small['ID'].unique().tolist(),
                                   palette=palette)


s1 = ColumnDataSource(df_small)
s2 = ColumnDataSource({'x':[df['time'].min(),df['time'].min()],'y':[-0.5,0.5]})
p = figure(plot_width=width, plot_height=height, title="Audio Waveform",
           toolbar_location='above', tools=[])
p.circle(x='time',y={'field':'id_adjusted','transform':Jitter(width=0.1)},color={'field':'ID','transform':color_map},source = s1,fill_alpha=0.01)
p.line(x='time',y = 'amplitude',source = s1)
l = p.line(x='x',y='y',source = s2,line_width=2,color='black')


d = Div()
play = False
toggle = Toggle(label='Play',active=False)
# noinspection PyUnresolvedReferences
def callback_tap(arg,s2=s2):
    xpos = arg.x
    data = s2.data
    #print(xpos)
    data['x']=[xpos,xpos]
    s2.data =data
    ap.seek(xpos)
    #s2.change.emit()

def callback_play(arg):
    #print(old)
  #  global play
   # play = not play
    p.title.text =f'{toggle.active}'
    if toggle.active:
        toggle.label = 'Pause'
        toggle.button_type='success'
        if not(ap.playing()):
            ap.play()
    else:
        toggle.label = 'Play'
        toggle.button_type = 'default'
        ap.pause()

def update():
    #global play
    #if play:
    if toggle.active:
        if s2.data['x'][0]> max_time:
            #play = False
            toggle.active = False
            s2.data['x'] = [0*i for i in s2.data['x']]
            d.text = f' Exceed time: {play}'
            ap.seek(s2.data['x'][0])
        else:
            d.text = f' Else: {play}'
            #s2.data['x'] = [i + dt/1000 for i in s2.data['x']]
            s2.data['x'] = [ap.time(),ap.time()]

            #s2.data['x'] = [s2.data['x'][0] + dt / 1000, s2.data['x'][0] + dt/1000]





curdoc().add_periodic_callback(update, dt)
p.on_event('tap',callback_tap)
toggle.on_click(callback_play)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(toggle, p,d))
