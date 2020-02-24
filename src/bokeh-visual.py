import os
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import column, row, grid
from bokeh.models import Div, ColumnDataSource, LinearColorMapper
from bokeh.models.widgets import Toggle
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, curdoc

from constants import width, height, dt
from utils.utils import pull_image_list, pull_speaker_id_time_df, generate_waveform_df

project_dir = Path(__file__).resolve().parents[1]

# Retrieve the args
args = curdoc().session_context.request.arguments
base_name = args.get("wavfile")[0].decode("utf-8")

fname = f"{project_dir}/data/raw/{base_name}.wav"
fname_rttm = f"{project_dir}/data/processed/{base_name}/{base_name}.rttm"
fname_speaker_id = f"{project_dir}/data/processed/{base_name}/{base_name}.pkl"
voxceleb_img_root = f"{project_dir}/data/images_small/"
filename_base = os.path.splitext(os.path.basename(fname))[0]

# Get ID vs time:
df_prob = pull_speaker_id_time_df(project_dir, fname_speaker_id)
# Get an image dictionary with celebrities
img_dict = pull_image_list(project_dir)

# Generate a dataframe with audiowaveform:
audio_player, df_waveform_full, max_time, df_waveform_small = generate_waveform_df(
    fname
)

source_waveform_small = ColumnDataSource(df_waveform_small)
source_waveform_full = ColumnDataSource(
    {
        "x": [df_waveform_full["time"].min(), df_waveform_full["time"].min()],
        "y": [-0.5, 0.5],
    }
)
bokeh_plot_waveform = figure(
    plot_width=width,
    plot_height=height,
    title="Sound waveform",
    toolbar_location="above",
    tools=[],
    output_backend="webgl",
)
bokeh_plot_waveform.line(x="time", y="amplitude", source=source_waveform_small)
line_waveform = bokeh_plot_waveform.line(
    x="x", y="y", source=source_waveform_full, line_width=2, color="black"
)

info_div = Div()
play = False
play_toggle = Toggle(label="Play", active=False)
record_toggle = Toggle(label="Record", active=False)


# noinspection PyUnresolvedReferences
def callback_tap(arg, s2=source_waveform_full):
    xpos = arg.x
    data = s2.data
    data["x"] = [xpos, xpos]
    s2.data = data
    audio_player.seek(xpos)
    cur_index = max(np.searchsorted(times_speakers, xpos) - 1, 0)
    print(cur_index)
    data_new_df = df_prob_per_time[cur_index]
    img = img_dict[data_new_df["Speaker"].values[-1]]
    image_src.data["image"] = [img]
    data_new = data_new_df.to_dict(orient="list")
    source_prob.data = data_new
    # source_prob_instant.data = df_prob_instant[cur_index].to_dict(orient='list')
    fctrs = source_prob.data["Speaker"]
    p_prob.x_range.factors = sorted(fctrs)


def callback_play(arg):
    bokeh_plot_waveform.title.text = f"Sound waveform"
    if play_toggle.active:
        play_toggle.label = "Pause"
        play_toggle.button_type = "success"
        if not (audio_player.playing()):
            audio_player.play()
    else:
        play_toggle.label = "Play"
        play_toggle.button_type = "default"
        audio_player.pause()


def callback_record(arg):
    bokeh_plot_waveform.title.text = f"Sound waveform"
    if play_toggle.active:
        play_toggle.label = "Recording.."
        play_toggle.button_type = "warning"
    else:
        play_toggle.label = "Record via mic"
        play_toggle.button_type = "default"


def update():
    global source_prob
    global times_speakers
    global cur_index
    if play_toggle.active:
        if source_waveform_full.data["x"][0] > max_time:
            # play = False
            play_toggle.active = False
            source_waveform_full.data["x"] = [
                0 * i for i in source_waveform_full.data["x"]
            ]
            audio_player.seek(source_waveform_full.data["x"][0])
            cur_index = 0
            data_new_df = df_prob_per_time[cur_index]
            cur_index += 1
            img = img_dict[data_new_df["Speaker"].values[-1]]
            image_src.data["image"] = [img]
            data_new = data_new_df.to_dict(orient="list")
            source_prob.data = data_new
            # source_prob_instant.data = df_prob_instant[cur_index].to_dict(orient='list')
            fctrs = source_prob.data["Speaker"]
            p_prob.x_range.factors = sorted(fctrs)
        else:
            source_waveform_full.data["x"] = [audio_player.time(), audio_player.time()]
            if len(times_speakers) > cur_index + 1:
                if audio_player.time() > times_speakers[cur_index]:
                    # d.text = f'Made a step to index {cur_index} , time of the next step = {times_speakers[cur_index]} '

                    # data_new_df = df_prob[df_prob.index <= ap.time()].groupby('Speaker').last().sort_values(
                    # 'cumsum_prob').reset_index()
                    data_new_df = df_prob_per_time[cur_index]
                    cur_index += 1
                    img = img_dict[data_new_df["Speaker"].values[-1]]
                    image_src.data["image"] = [img]
                    data_new = data_new_df.to_dict(orient="list")
                    source_prob.data = data_new
                    # source_prob_instant.data = df_prob_instant[cur_index].to_dict(orient='list')
                    fctrs = source_prob.data["Speaker"]
                    p_prob.x_range.factors = list(fctrs)


df_prob["cumsum_prob"] = df_prob.groupby(["Speaker"])["Probability"].transform(
    pd.Series.cumsum
)
times_speakers = df_prob.index.unique().values
df_prob_per_time = {}
df_prob_instant = {}
for t in times_speakers:
    tmp = (
        df_prob[df_prob.index <= t]
        .groupby("Speaker")
        .last()
        .sort_values("cumsum_prob")
        .reset_index()
        .tail(10)
    )
    tmp["cumsum_prob"] = tmp["cumsum_prob"] / tmp["cumsum_prob"].sum()
    df_prob_per_time[t] = tmp
    df_prob_instant[t] = df_prob.loc[t, :]

mapper = LinearColorMapper(palette=Spectral10, low=0, high=df_prob.index.nunique())

df_subset = (
    df_prob[df_prob.index < 0]
    .groupby("Speaker")
    .last()
    .sort_values("cumsum_prob")
    .reset_index()
)
cur_index = 1

source_prob = ColumnDataSource(data=df_subset)
(xdim, ydim) = img_dict["Empty"].shape
image_src = ColumnDataSource(data={"image": [np.flipud(img_dict["Empty"])]})  # OK
p_face = figure(
    plot_height=int(1.5 * height),
    plot_width=int(width / 3),
    x_range=(0, xdim),
    y_range=(0, ydim),
    tools=[],
)
p_face.axis.visible = False

tt = [("Speaker:", "@Speaker"), ("Prob", "@cumsum_prob")]
p_prob = figure(
    x_range=source_prob.data["Speaker"],
    plot_width=int(2 * width / 3),
    plot_height=int(1.5 * height),
    tools=[],
    tooltips=tt,
)
p_prob.xaxis.major_label_orientation = 3.14 / 4
p_prob.vbar(
    x="Speaker",
    top="cumsum_prob",
    fill_color={"field": "cumsum_prob", "transform": mapper},
    width=0.5,
    source=source_prob,
)

p_face.image("image", x=0, y=0, dw=xdim, dh=ydim, source=image_src)

curdoc().add_periodic_callback(update, dt)
bokeh_plot_waveform.on_event("tap", callback_tap)
play_toggle.on_click(callback_play)
# put the button and plot in a layout and add to the document
curdoc().add_root(grid(column(play_toggle, bokeh_plot_waveform, row(p_prob, p_face))))
