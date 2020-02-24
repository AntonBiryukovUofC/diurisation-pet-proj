import pandas as pd
import numpy as np
import pyannote.audio
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from pandas.api.types import CategoricalDtype
from src.player import AudioPlayer
from src.viewer import PlotDiar
import seaborn as sns
from tqdm import trange

fname = "data/raw/z-c-feisty.wav"
filename_base = os.path.splitext(os.path.basename(fname))[0]
x, sr = librosa.load(fname)
print(filename_base)

fname_rttm = "data/processed/test.rttm"
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
cat_type = CategoricalDtype(categories=speaker_track["speaker_id"].unique().tolist())

df = (
    pd.DataFrame({"time": np.arange(x.shape[0]) / sr, "amplitude": x})
    .sample(frac=0.001)
    .sort_values("time")
)

p = PlotDiar(
    wav=fname,
    gui=True,
    size=(15, 6),
    maxx=df["time"].max(),
    df_tracks=df,
    df_speakers=speaker_track,
)
p.draw()
p.plot.show()

#
# df = pd.DataFrame({"time": np.arange(x.shape[0]) / sr, "amplitude": x})
# def find_speaker(x):
#     try:
#         res = bins_speaker.loc[x, 'name'].values[0]
#     except:
#         res = np.nan
#     return res
# tmp = df.sample(frac=0.005).sort_values('time')
# speaker_id = tmp['time'].apply(lambda x: find_speaker(x))
# speaker_id.head()
# tmp['speaker'] = speaker_id.astype(cat_type)
# tmp.set_index('time', inplace=True, drop=False)
# folder_output = os.path.abspath(os.path.splitext(fname)[0])
# folder_output_ff = os.path.abspath(os.path.splitext(fname)[0])
# os.makedirs(folder_output, exist_ok=True)
# # fig,ax = plt.subplots(figsize=(27, 5),nrows=2,gridspec_kw = {'height_ratios':[3,2]})
# fig, ax = plt.subplots(figsize=(14, 2), nrows=1)
# tmp['speaker_cat'] = tmp['speaker'].cat.codes
# tmp.loc[tmp['speaker_cat'] == -1, 'speaker_cat'] = np.NaN
# sns.stripplot(x='time', y='speaker', data=tmp, ax=ax, alpha=0.05, palette='Set2', jitter=0.4, size=6.5, zorder=0)
# tmp['amp_scaled'] = np.mean(ax.get_ylim()) + tmp['amplitude'] * len(tmp['speaker'].unique())
# tmp['amp_scaled'].plt(ax=ax, zorder=10)
# sec = 0
# dsec = 15
# # line = ax[1].axvline(sec,c = 'k',linewidth=5)
# line = ax.axvline(sec, c='k', linewidth=5)
# line_l = ax.axvline(sec - dsec * 0.95, c='k', alpha=0.2, linewidth=5)
# line_r = ax.axvline(sec + dsec * 0.95, c='k', alpha=0.2, linewidth=5)
# n_frames = 337
# sec_per_frame = (df['time'].max() + 1) / n_frames
# ax.set_xlim([0, df['time'].max() + 1])
# frame_rate = x.shape[0] / sr / (n_frames + 1)
# tmp_file_ffmpeg = ['ffconcat version 1.0\n']
