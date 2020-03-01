import os

import librosa
import numpy as np
import pandas as pd
from PIL import Image

from player import AudioPlayer
from vggvoxvlad.split import run_split
import pathos.multiprocessing as mp


def initialize_GPU(args):
    # Initialize GPUs
    import tensorflow as tf

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i : i + n]


# set up multiprocessing
def set_mp(processes=8):
    # import multiprocessing as mp

    def init_worker():
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.ProcessingPool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


# vggface2 dataset
def get_vggface2_imglist(args):
    def get_datalist(s):
        file = open("{}".format(s), "r")
        datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(" ")
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1][:-1]))
        return imglist, labellist

    print("==> calculating image lists...")
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val


def get_imagenet_imglist(args, trn_meta_path="", val_meta_path=""):
    with open(trn_meta_path) as f:
        strings = f.readlines()
        trn_list = np.array(
            [
                os.path.join(
                    args.data_path, "/".join(string.split()[0].split(os.sep)[-4:])
                )
                for string in strings
            ]
        )
        trn_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()

    with open(val_meta_path) as f:
        strings = f.readlines()
        val_list = np.array(
            [
                os.path.join(
                    args.data_path, "/".join(string.split()[0].split(os.sep)[-4:])
                )
                for string in strings
            ]
        )
        val_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return trn_list, trn_lb, val_list, val_lb


def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array(
            [os.path.join(args.data_path, string.split()[0]) for string in strings]
        )
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist


def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def sync_model(src_model, tgt_model):
    print("==> synchronizing the model weights.")
    params = {}
    for l in src_model.layers:
        params["{}".format(l.name)] = l.get_weights()

    for l in tgt_model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights(params["{}".format(l.name)])
    return tgt_model


def pull_image_list(project_dir,voxceleb_img_root,df_prob):
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
        img_dict[i] = np.flipud(np.asarray(im)[:, :, 0])
    img_dict["Empty"] = img_dict[list(img_dict.keys())[0]].copy()
    np.random.shuffle(img_dict["Empty"])
    return img_dict


def pull_speaker_id_time_df(project_dir, fname_speaker_id,fname,base_name):
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
        df_prob = pd.read_pickle(fname_speaker_id).set_index("Time_s")
    else:
        # Preload speaker ID vs time dataframes
        df_prob = pd.read_pickle(fname_speaker_id).set_index("Time_s")
    return df_prob


def generate_waveform_df(fname):
    x, sr = librosa.load(fname)
    ap = AudioPlayer(fname)
    df = pd.DataFrame({"time": np.arange(x.shape[0]) / (sr), "amplitude": x})
    max_time = df["time"].max()
    df_small = df.sample(frac=0.01).sort_values("time")
    df_small["amplitude"] = (df_small["amplitude"] - df_small["amplitude"].min()) / (
        df_small["amplitude"].max() - df_small["amplitude"].min()
    ) - 0.5
    return ap, df, max_time, df_small
