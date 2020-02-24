import librosa
import pandas as pd
from .utils_dan import load_data
import numpy as np
from .model import vggvox_resnet2d_icassp
import matplotlib.pyplot as plt
from tool.toolkits import initialize_GPU


def make_network(weight_path, args, input_dim=(257, None, 1), num_class=1251):
    network_eval = vggvox_resnet2d_icassp(
        input_dim=input_dim, num_class=num_class, mode="eval", args=args
    )
    network_eval.load_weights(weight_path, by_name=True)

    return network_eval


def voxceleb1_split(
    path,
    network,
    split_seconds=3,
    shift_seconds=1.5,
    n_top_candidates=3,
    win_length=400,
    sr=16000,
    hop_length=160,
    n_fft=512,
    spec_len=250,
    n_classes=1251,
    metafile_location="../data/raw/vox1_meta.txt",
):
    from tqdm import trange

    vc_df = pd.read_csv(metafile_location, sep="\t", skiprows=0)
    vc_df["class"] = pd.to_numeric(vc_df["VoxCeleb1 ID"].str.replace("id", "")) - 10001

    x, sr = librosa.load(path, sr)

    df_audio = pd.DataFrame({"Time": np.arange(x.shape[0]) / sr, "Amplitude": x})

    num_segments = int(
        np.floor((df_audio["Time"].max() - split_seconds / 2) / shift_seconds)
    )

    result_list = []
    t = 0
    dt = split_seconds  # how often to split audio file for predicting

    for k in trange(num_segments):

        df_out = pd.DataFrame([])

        df_tmp = df_audio.copy()
        df_tmp = df_tmp[(df_tmp["Time"] >= t) & (df_tmp["Time"] < (t + dt))]

        amp = np.array(df_tmp["Amplitude"])

        specs = load_data(
            amp,
            win_length=win_length,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            spec_len=spec_len,
            mode="eval",
        )
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)

        v = network.predict(specs)

        # find top 3
        v = v.reshape(n_classes)
        ind = (-v).argsort()[:n_top_candidates]

        for i in range(n_top_candidates):
            df_out = df_out.append(
                pd.DataFrame(
                    {
                        "Time_s": t,
                        "Speaker": vc_df["VGGFace1 ID"][ind[i]].replace("_", " "),
                        "Probability": v[ind][i],
                        "Country": vc_df["Nationality"][ind[i]],
                        "Gender": vc_df["Gender"][ind[i]],
                    },
                    index=[0],
                ),
                ignore_index=True,
            )
        # df_out = df_out.set_index('Time (s)')

        result_list.append(df_out)

        t += shift_seconds

    return result_list


def plot_split(result_list, num_speakers=2):
    t = []
    prob = []
    name = []

    for df in result_list:
        if df["Probability"][0] >= 0.5:
            t.append(df["Time (s)"][0])
            prob.append(df["Probability"][0])
            name.append(df["Speaker"][0])
    t = np.array(t)
    prob = np.array(prob)

    from collections import Counter

    speaker_dict = Counter(name)
    sorted_speaker = sorted(speaker_dict.items(), key=lambda kv: kv[1])
    sorted_speaker = sorted_speaker[-num_speakers:]
    names = [i[0] for i in sorted_speaker]

    for i in range(len(name)):
        if name[i] not in names:
            name[i] = "Other"

    import seaborn as sns

    plt.figure(figsize=(16, 2))
    by_school = sns.barplot(x=t, y=prob, hue=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")

    for item in by_school.get_xticklabels():
        item.set_rotation(90)


def run_split(
    weight_path="../models/vggvox/weights-09-0.923.h5",
    fname="data/raw/atkinson-clarkson.wav",
    metafile_location="data/raw/vox1_meta.txt",
    split_seconds=3,
    shift_seconds=1,
):
    import os
    import argparse
    import pandas as pd

    # gpu configuration

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--resume", default=r"pretrained/weights.h5", type=str)
    parser.add_argument("--data_path", default="4persons", type=str)
    # set up network configuration.
    parser.add_argument(
        "--net", default="resnet34s", choices=["resnet34s", "resnet34l"], type=str
    )
    parser.add_argument("--ghost_cluster", default=2, type=int)
    parser.add_argument("--vlad_cluster", default=8, type=int)
    parser.add_argument("--bottleneck_dim", default=512, type=int)
    parser.add_argument(
        "--aggregation_mode",
        default="gvlad",
        choices=["avg", "vlad", "gvlad"],
        type=str,
    )
    # set up learning rate, training loss and optimizer.
    parser.add_argument(
        "--loss", default="softmax", choices=["softmax", "amsoftmax"], type=str
    )
    parser.add_argument(
        "--test_type", default="normal", choices=["normal", "hard", "extend"], type=str
    )

    args = parser.parse_args(args=[])
    initialize_GPU(args)
    net = make_network(weight_path, args)
    basename = os.path.splitext(os.path.basename(fname))[0]
    output_folder = f"data/processed/{basename}"

    os.makedirs(output_folder, exist_ok=True)
    output_filename_pkl = os.path.join(output_folder, f"{basename}.pkl")
    output_filename_csv = os.path.join(output_folder, f"{basename}.csv")

    result_list = voxceleb1_split(
        path=fname,
        network=net,
        metafile_location=metafile_location,
        split_seconds=split_seconds,
        shift_seconds=shift_seconds,
    )
    result_df = pd.concat(result_list)

    # print(f'PKL saving under {output_filename_pkl}')
    # print(f'CSV saving under {output_filename_csv}')
    #
    # result_df.to_pickle(output_filename_pkl)
    # result_df.to_csv(output_filename_csv)
    return result_df
