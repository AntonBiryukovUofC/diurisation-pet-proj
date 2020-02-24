import sys
import argparse
import numpy as np

sys.path.insert(0, ".")
# gpu configuration
import matplotlib.pyplot as plt
import src.vggvoxvlad.model as model
import src.tool.toolkits as toolkits
import src.vggvoxvlad.utils as ut

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

network_eval = model.vggvox_resnet2d_icassp(
    input_dim=params["dim"], num_class=params["n_classes"], mode="eval", args=args
)

network_eval.load_weights("models/vggvox/weights-09-0.923.h5", by_name=True)

ID = r"D:/VoxCeleb/vox1_dev_wav/wav/id10751/aacBLJr-hs0/00004.wav"
specs = ut.load_data(
    ID,
    win_length=params["win_length"],
    sr=params["sampling_rate"],
    hop_length=params["hop_length"],
    n_fft=params["nfft"],
    spec_len=params["spec_len"],
    mode="eval",
)
specs = np.expand_dims(np.expand_dims(specs, 0), -1)

v = network_eval.predict(specs)
# plt.plot(v[0,:]);
print(f"Predicted ID: {v.argmax()+1}")
