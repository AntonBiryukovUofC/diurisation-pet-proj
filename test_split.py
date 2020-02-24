import pandas as pd
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, "/Users/danielsola/Downloads/diurisation-pet-proj-master_3/")
# gpu configuration
import src.vggvoxvlad.split as splt
import matplotlib.pyplot as plt
import src.vggvoxvlad.model as model
import src.utils.utils as toolkits
import src.vggvoxvlad.utils_dan as ut_d
import src.vggvoxvlad.utils as ut
import numpy as np


args = []

parser = argparse.ArgumentParser()
# set up training configuration.
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
    "--aggregation_mode", default="gvlad", choices=["avg", "vlad", "gvlad"], type=str
)
# set up learning rate, training loss and optimizer.
parser.add_argument(
    "--loss", default="softmax", choices=["softmax", "amsoftmax"], type=str
)
parser.add_argument(
    "--test_type", default="normal", choices=["normal", "hard", "extend"], type=str
)

args = parser.parse_args(args=[])
toolkits.initialize_GPU(args)
params = {
    "dim": (257, None, 1),
    "nfft": 512,
    "spec_len": 250,
    "win_length": 400,
    "hop_length": 160,
    "n_classes": 1251,
    "sampling_rate": 16000,
    "normalize": True,
}

weight_path = "models/vggvox/weights-09-0.923.h5"
net = splt.make_network(weight_path, args)
fname = "data/raw/atkinson-clarkson.wav"
basename = os.path.splitext(os.path.basename(fname))[0]
output_folder = f"data/processed/{basename}"

os.makedirs(output_folder, exist_ok=True)
output_filename_pkl = os.path.join(output_folder, f"{basename}.pkl")
output_filename_csv = os.path.join(output_folder, f"{basename}.csv")


result_list = splt.voxceleb1_split(
    path=fname,
    network=net,
    metafile_location="data/raw/vox1_meta.txt",
    split_seconds=3,
    shift_seconds=1,
)
result_df = pd.concat(result_list)
print(f"PKL saving under {output_filename_pkl}")
print(f"CSV saving under {output_filename_csv}")

result_df.to_pickle(output_filename_pkl)
result_df.to_csv(output_filename_csv)
