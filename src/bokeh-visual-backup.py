from bokeh.plotting import figure, show, curdoc
from bokeh.events import Tap
from bokeh.models.callbacks import CustomJS
from bokeh.models import (
    ColorBar,
    LinearColorMapper,
    Div,
    HoverTool,
    ColumnDataSource,
    TapTool,
)
import sys
import time

sys.path.insert(0, "D:\\Repos\\diurisation-pet-proj")
from bokeh.io import save
from bokeh.client import push_session
from bokeh.models.widgets import Toggle
from src.player import AudioPlayer
from bokeh.layouts import row, column
from bokeh.palettes import d3
from bokeh.palettes import RdBu11, OrRd9, Blues9
from bokeh.transform import linear_cmap
from bokeh.models import CategoricalColorMapper
import librosa
import os
import numpy as np
import pandas as pd


def load_rttm(rttm_file_name):
    speaker_track = pd.read_csv(
        fname_rttm,
        header=None,
        delim_whitespace=True,
        names=[
            "x1",
            "filename",
            "fileid",
            "start",
            "duration",
            "skip1",
            "skip2",
            "speaker_id",
            "skip3",
        ],
    )
    speaker_track["end"] = speaker_track["start"] + speaker_track["duration"]
    speaker_track["id"] = speaker_track["speaker_id"].apply(
        lambda x: int(x.split("speaker")[1])
    )

    bins_speaker = pd.DataFrame(
        {"name": speaker_track["speaker_id"].values},
        index=pd.IntervalIndex.from_arrays(
            left=speaker_track["start"], right=speaker_track["end"]
        ),
    )
    speaker_track.drop(["skip1", "skip2", "skip3"], axis=1, inplace=True)
    speaker_track = speaker_track[
        speaker_track["duration"] > 0.25
    ]  # Keep only those longer than 0.25 secs
    return speaker_track
    # cat_type = CategoricalDtype(categories=speaker_track['speaker_id'].unique().tolist())


fname = "../data/raw/carell.wav"
fname_rttm = "../data/processed/carell/carell.rttm"
rttm_df = load_rttm(fname_rttm)
print(rttm_df.head())
filename_base = os.path.splitext(os.path.basename(fname))[0]
x, sr = librosa.load(fname)
ap = AudioPlayer(fname)
fr = ap.wf.getframerate()
df = pd.DataFrame({"time": np.arange(x.shape[0]) / (sr), "amplitude": x})
max_time = df["time"].max()
df_small = df.sample(frac=0.05).sort_values("time")
df_small["amplitude"] = (df_small["amplitude"] - df_small["amplitude"].min()) / (
    df_small["amplitude"].max() - df_small["amplitude"].min()
) - 0.5
dt = 50  # in ms


s1 = ColumnDataSource(df_small)
s2 = ColumnDataSource({"x": [df["time"].min(), df["time"].min()], "y": [-0.5, 0.5]})
p = figure(
    plot_width=700,
    plot_height=150,
    title="Audio Waveform",
    toolbar_location="above",
    tools=[],
)


p.line(x="time", y="amplitude", source=s1)
l = p.line(x="x", y="y", source=s2, line_width=2, color="black")


d = Div()
play = False
toggle = Toggle(label="Play", active=False)
# noinspection PyUnresolvedReferences
def callback_tap(arg, s2=s2):
    xpos = arg.x
    data = s2.data
    # print(xpos)
    data["x"] = [xpos, xpos]
    s2.data = data
    ap.seek(xpos)
    # s2.change.emit()


def callback_play(arg):
    # print(old)
    #  global play
    # play = not play
    p.title.text = f"{toggle.active}"
    if toggle.active:
        toggle.label = "Pause"
        toggle.button_type = "success"
        if not (ap.playing()):
            ap.play()
    else:
        toggle.label = "Play"
        toggle.button_type = "default"
        ap.pause()


def update():
    # global play
    # if play:
    if toggle.active:
        if s2.data["x"][0] > max_time:
            # play = False
            toggle.active = False
            s2.data["x"] = [0 * i for i in s2.data["x"]]
            d.text = f" Exceed time: {play}"
            ap.seek(s2.data["x"][0])
        else:
            d.text = f" Else: {play}"
            # s2.data['x'] = [i + dt/1000 for i in s2.data['x']]
            s2.data["x"] = [ap.time(), ap.time()]

            # s2.data['x'] = [s2.data['x'][0] + dt / 1000, s2.data['x'][0] + dt/1000]


curdoc().add_periodic_callback(update, dt)
p.on_event("tap", callback_tap)
toggle.on_click(callback_play)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(toggle, p, d))
