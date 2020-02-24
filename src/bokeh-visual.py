import sys

from bokeh.models import Div, ColumnDataSource, LinearColorMapper
from bokeh.palettes import Spectral10, Category10
from bokeh.plotting import figure, curdoc
from vggvoxvlad.split import run_split
from bokeh.models.widgets import Toggle
from player import AudioPlayer
from bokeh.layouts import column, row, grid
import librosa
from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image

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
print('Called bokeh visual')
width = 1000
height = 250
# Retrieve the args
args = curdoc().session_context.request.arguments

print(args)

project_dir = Path(__file__).resolve().parents[1]

base_name = args.get("wavfile")[0].decode("utf-8")

fname = f"{project_dir}/data/raw/{base_name}.wav"
fname_rttm = f"{project_dir}/data/processed/{base_name}/{base_name}.rttm"
fname_speaker_id = f"{project_dir}/data/processed/{base_name}/{base_name}.pkl"
voxceleb_img_root = f"{project_dir}/data/images_small/"
# Check if pickle exists - if it does not, then create one!
if not (os.path.isfile(fname_speaker_id)):
    result_df = run_split(
        weight_path=f"{project_dir}/models/vggvox/weights-09-0.923.h5",
        fname=fname,
        metafile_location=f"{project_dir}/data/raw/vox1_meta.txt",
        split_seconds=3,
        shift_seconds=1,
    )
    output_folder = f"{project_dir}/data/processed/{base_name}"
    os.makedirs(output_folder, exist_ok=True)
    result_df.to_pickle(fname_speaker_id)

# Preload RTTM
# rttm_df = load_rttm(fname_rttm, min_duration=0.2)
# Preload speaker ID vs time dataframes
df_prob = pd.read_pickle(fname_speaker_id).set_index("Time_s")

# Preload images from VoxCeleb:
images_list = pd.read_csv(f"{project_dir}/data/raw/image_list.csv")
images_list.set_index("Celeb Name", inplace=True)
il_all = (
    images_list.reset_index()
    .groupby("Celeb Name")
    .agg(pd.DataFrame.sample)
    .loc[df_prob["Speaker"]]
)
img_dict = {}
# size = 128, 128
for i, r in il_all.iterrows():
    fname_image = os.path.join(voxceleb_img_root, r["File Name"])
    im = Image.open(fname_image).convert("RGB")
    # im.thumbnail(size)
    img_dict[i] = np.flipud(np.asarray(im)[:, :, 0])
# Fu
img_dict["Empty"] = img_dict[list(img_dict.keys())[0]].copy()
np.random.shuffle(img_dict["Empty"])

filename_base = os.path.splitext(os.path.basename(fname))[0]
x, sr = librosa.load(fname)
ap = AudioPlayer(fname)
fr = ap.wf.getframerate()
df = pd.DataFrame({"time": np.arange(x.shape[0]) / (sr), "amplitude": x})
max_time = df["time"].max()
df_small = df.sample(frac=0.001).sort_values("time")
df_small["amplitude"] = (df_small["amplitude"] - df_small["amplitude"].min()) / (
    df_small["amplitude"].max() - df_small["amplitude"].min()
) - 0.5
# df_small = label_waveform_by_speaker(df_small, rttm_df)
dt = 200  # in ms
print(df_small.sample(frac=1).head())
print(df_small.shape)

# palette = d3['Category10'][len(df_small['ID'].unique())]
# color_map = CategoricalColorMapper(factors=df_small['ID'].unique().tolist(),
#                                   palette=palette)

s1 = ColumnDataSource(df_small)
s2 = ColumnDataSource({"x": [df["time"].min(), df["time"].min()], "y": [-0.5, 0.5]})
p = figure(
    plot_width=width,
    plot_height=height,
    title="Sound waveform",
    toolbar_location="above",
    tools=[],
    output_backend="webgl",
)
# p.circle(x='time',y={'field':'id_adjusted','transform':Jitter(width=0.1)},color={'field':'ID','transform':color_map},source = s1,fill_alpha=0.01)
p.line(x="time", y="amplitude", source=s1)
l = p.line(x="x", y="y", source=s2, line_width=2, color="black")

d = Div()
play = False
toggle = Toggle(label="Play", active=False)
toggle_record = Toggle(label="Record", active=False)


# noinspection PyUnresolvedReferences
def callback_tap(arg, s2=s2):
    xpos = arg.x
    data = s2.data
    data["x"] = [xpos, xpos]
    s2.data = data
    ap.seek(xpos)
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
    p.title.text = f"Sound waveform"
    if toggle.active:
        toggle.label = "Pause"
        toggle.button_type = "success"
        if not (ap.playing()):
            ap.play()
    else:
        toggle.label = "Play"
        toggle.button_type = "default"
        ap.pause()


def callback_record(arg):
    p.title.text = f"Sound waveform"
    if toggle.active:
        toggle.label = "Recording.."
        toggle.button_type = "warning"
    else:
        toggle.label = "Record via mic"
        toggle.button_type = "default"


def update():
    global source_prob
    global times_speakers
    global cur_index
    if toggle.active:
        if s2.data["x"][0] > max_time:
            # play = False
            toggle.active = False
            s2.data["x"] = [0 * i for i in s2.data["x"]]
            ap.seek(s2.data["x"][0])
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
            s2.data["x"] = [ap.time(), ap.time()]
            if len(times_speakers) > cur_index + 1:
                if ap.time() > times_speakers[cur_index]:
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


##### Probability figure section!
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
# d.text = f' Index {cur_index} , time of the next step = {times_speakers[cur_index]} '


source_prob = ColumnDataSource(data=df_subset)
# source_prob_instant = ColumnDataSource(data=df_prob_instant[0])

img = np.flipud(img_dict["Empty"])
(xdim, ydim) = img.shape
image_src = ColumnDataSource(data={"image": [img]})  # OK
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

# p_prob.vbar(x='Speaker', top='Probability', fill_color=Category10[3][2],
#             source=source_prob_instant,fill_alpha=0.8,width=0.3)
p_face.image("image", x=0, y=0, dw=xdim, dh=ydim, source=image_src)

####


curdoc().add_periodic_callback(update, dt)
p.on_event("tap", callback_tap)
toggle.on_click(callback_play)
# put the button and plot in a layout and add to the document
curdoc().add_root(grid(column(toggle, p, row(p_prob, p_face))))
