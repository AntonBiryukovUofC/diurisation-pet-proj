import configparser
import os

import numpy as np
import torch
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.signal import Binarize

from src.pybk.diarizationFunctions import (
    extractFeatures,
    readUEMfile,
    readSADfile,
    getSegmentTable,
    trainKBM,
    getVgMatrix,
    getSegmentBKs,
    performClustering,
    performClusteringLinkage,
    getBestClustering,
    getSpectralClustering,
    performResegmentation,
    getSegmentationFile,
)


def run_diarization(showName, config, sad_file_name):
    print("showName\t\t", showName)
    print("Extracting features")
    all_data = extractFeatures(
        config["PATH"]["audio"] + showName + config["EXTENSION"]["audio"],
        config.getfloat("FEATURES", "framelength"),
        config.getfloat("FEATURES", "frameshift"),
        config.getint("FEATURES", "nfilters"),
        config.getint("FEATURES", "ncoeff"),
    )
    n_features = all_data.shape[0]
    print("Initial number of features\t", n_features)
    if os.path.isfile(config["PATH"]["UEM"] + showName + config["EXTENSION"]["UEM"]):
        mask_uem = readUEMfile(
            config["PATH"]["UEM"],
            showName,
            config["EXTENSION"]["UEM"],
            n_features,
            config.getfloat("FEATURES", "frameshift"),
        )
    else:
        print("UEM file does not exist. The complete audio content is considered.")
        mask_uem = np.ones([1, n_features])
    mask_sad = readSADfile(
        sad_file_name,
        n_features,
        config.getfloat("FEATURES", "frameshift"),
        config["GENERAL"]["SADformat"],
    )
    mask = np.logical_and(mask_uem, mask_sad)
    mask = mask[0][0:n_features]
    n_speech_features = np.sum(mask)
    speech_mapping = np.zeros(n_features)
    # you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
    # so that we don't lose features on the way
    speech_mapping[np.nonzero(mask)] = np.arange(1, n_speech_features + 1)
    data = all_data[np.where(mask == 1)]
    del all_data
    segment_table = getSegmentTable(
        mask,
        speech_mapping,
        config.getint("SEGMENT", "length"),
        config.getint("SEGMENT", "increment"),
        config.getint("SEGMENT", "rate"),
    )
    numberOfSegments = np.size(segment_table, 0)
    print("Number of speech features\t", n_speech_features)
    # create the KBM
    print("Training the KBM... ")
    # set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
    if np.floor(
        (n_speech_features - config.getint("KBM", "windowLength"))
        / config.getint("KBM", "minimumNumberOfInitialGaussians")
    ) < config.getint("KBM", "maximumKBMWindowRate"):
        window_rate = int(
            np.floor(
                (np.size(data, 0) - config.getint("KBM", "windowLength"))
                / config.getint("KBM", "minimumNumberOfInitialGaussians")
            )
        )
    else:
        window_rate = int(config.getint("KBM", "maximumKBMWindowRate"))
    pool_size = np.floor(
        (n_speech_features - config.getint("KBM", "windowLength")) / window_rate
    )
    if config.getint("KBM", "useRelativeKBMsize"):
        kbm_size = int(np.floor(pool_size * config.getfloat("KBM", "relKBMsize")))
    else:
        kbm_size = int(config.getint("KBM", "kbmSize"))
    print(
        "Training pool of",
        int(pool_size),
        "gaussians with a rate of",
        int(window_rate),
        "frames",
    )
    kbm, gm_pool = trainKBM(
        data, config.getint("KBM", "windowLength"), window_rate, kbm_size
    )
    print("Selected", kbm_size, "gaussians from the pool")
    vg = getVgMatrix(
        data, gm_pool, kbm, config.getint("BINARY_KEY", "topGaussiansPerFrame")
    )
    print("Computing binary keys for all segments... ")
    segment_bk_table, segment_cv_table = getSegmentBKs(
        segment_table,
        kbm_size,
        vg,
        config.getfloat("BINARY_KEY", "bitsPerSegmentFactor"),
        speech_mapping,
    )
    print("Performing initial clustering... ")
    initial_clustering = np.digitize(
        np.arange(numberOfSegments),
        np.arange(
            0,
            numberOfSegments,
            numberOfSegments / config.getint("CLUSTERING", "N_init"),
        ),
    )
    print("done")
    print("Performing agglomerative clustering... ")
    if config.getint("CLUSTERING", "linkage"):
        final_clustering_table, k = performClusteringLinkage(
            segment_bk_table,
            segment_cv_table,
            config.getint("CLUSTERING", "N_init"),
            config["CLUSTERING"]["linkageCriterion"],
            config["CLUSTERING"]["metric"],
        )
    else:
        final_clustering_table, k = performClustering(
            speech_mapping,
            segment_table,
            segment_bk_table,
            segment_cv_table,
            vg,
            config.getfloat("BINARY_KEY", "bitsPerSegmentFactor"),
            kbm_size,
            config.getint("CLUSTERING", "N_init"),
            initial_clustering,
            config["CLUSTERING"]["metric"],
        )
    print("Selecting best clustering...")
    if config["CLUSTERING_SELECTION"]["bestClusteringCriterion"] == "elbow":
        best_clustering_id = getBestClustering(
            config["CLUSTERING_SELECTION"]["metric_clusteringSelection"],
            segment_bk_table,
            segment_cv_table,
            final_clustering_table,
            k,
        )
    elif config["CLUSTERING_SELECTION"]["bestClusteringCriterion"] == "spectral":
        best_clustering_id = (
            getSpectralClustering(
                config["CLUSTERING_SELECTION"]["metric_clusteringSelection"],
                final_clustering_table,
                config.getint("CLUSTERING", "N_init"),
                segment_bk_table,
                segment_cv_table,
                k,
                config.getint("CLUSTERING_SELECTION", "sigma"),
                config.getint("CLUSTERING_SELECTION", "percentile"),
                config.getint("CLUSTERING_SELECTION", "maxNrSpeakers"),
            )
            + 1
        )
    print("Best clustering:\t", best_clustering_id.astype(int))
    print(
        "Number of clusters:\t",
        np.size(
            np.unique(final_clustering_table[:, best_clustering_id.astype(int) - 1]), 0
        ),
    )
    if (
        config.getint("RESEGMENTATION", "resegmentation")
        and np.size(
            np.unique(final_clustering_table[:, best_clustering_id.astype(int) - 1]), 0
        )
        > 1
    ):
        print("Performing GMM-ML resegmentation...")
        final_clustering_table_resegmentation, final_segment_table = performResegmentation(
            data,
            speech_mapping,
            mask,
            final_clustering_table[:, best_clustering_id.astype(int) - 1],
            segment_table,
            config.getint("RESEGMENTATION", "modelSize"),
            config.getint("RESEGMENTATION", "nbIter"),
            config.getint("RESEGMENTATION", "smoothWin"),
            n_speech_features,
        )
        print("done")
        output_fname = getSegmentationFile(
            config["OUTPUT"]["format"],
            config.getfloat("FEATURES", "frameshift"),
            final_segment_table,
            np.squeeze(final_clustering_table_resegmentation),
            showName,
            config["EXPERIMENT"]["name"],
            config["PATH"]["output"],
            config["EXTENSION"]["output"],
        )
    else:
        output_fname = getSegmentationFile(
            config["OUTPUT"]["format"],
            config.getfloat("FEATURES", "frameshift"),
            segment_table,
            final_clustering_table[:, best_clustering_id.astype(int) - 1],
            showName,
            config["EXPERIMENT"]["name"],
            config["PATH"]["output"],
            config["EXTENSION"]["output"],
        )

    if config.getint("OUTPUT", "returnAllPartialSolutions"):
        if not os.path.isdir(config["PATH"]["output"]):
            os.mkdir(config["PATH"]["output"])
        output_path_ind = (
            config["PATH"]["output"]
            + config["EXPERIMENT"]["name"]
            + "/"
            + showName
            + "/"
        )
        if not os.path.isdir(config["PATH"]["output"] + config["EXPERIMENT"]["name"]):
            os.mkdir(config["PATH"]["output"] + config["EXPERIMENT"]["name"])
        if not os.path.isdir(output_path_ind):
            os.mkdir(output_path_ind)
        for i in np.arange(k):
            getSegmentationFile(
                config["OUTPUT"]["format"],
                config.getfloat("FEATURES", "frameshift"),
                segment_table,
                final_clustering_table[:, i],
                showName,
                showName
                + "_"
                + str(np.size(np.unique(final_clustering_table[:, i]), 0))
                + "_spk",
                output_path_ind,
                config["EXTENSION"]["output"],
            )

    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return output_fname


def generate_sad_lbl(
    filename_input=f"../data/raw/z-c-feisty.wav",
    output_folder=f"../data/sad/",
    onset=0.7,
    offset=0.7,
    model=None,
):
    basename = os.path.splitext(os.path.basename(filename_input))[0]
    output_filename = os.path.join(output_folder, f"{basename}.lbl")
    test_file = {"uri": "filename", "audio": filename_input}
    sad_scores = model(test_file)
    # binarize raw SAD scores (as `pyannote.core.Timeline` instance)
    # NOTE: both onset/offset values were tuned on AMI dataset.
    # you might need to use different values for better results.
    binarized_scores = Binarize(offset=offset, onset=onset, log_scale=True)
    speech = binarized_scores.apply(sad_scores, dimension=1)
    # iterate over speech segments (as `pyannote.core.Segment` instances)
    print(f"Size of speech segment container = {len(speech)}")
    speech_str = [f"{i.start} {i.end} speech" for i in speech]
    sad_content = "\n".join(speech_str)
    with open(output_filename, "w") as f:
        f.write(sad_content)
    return output_filename


def run_pyBK_diarisation(
    config_loc=None,
    input_file_name=None,
    input_file_folder=None,
    output_folder=None,
    output_name=None,
    sad_file_name=None,
):
    config = configparser.ConfigParser()
    config.read(config_loc)
    if output_folder is not None:
        config["PATH"]["output"] = output_folder
    if sad_file_name is not None:
        config["PATH"]["SAD"] = ""
    if output_name is not None:
        config["EXPERIMENT"]["name"] = output_name
    if input_file_folder is not None:
        config["PATH"]["audio"] = input_file_folder
    # Audio files are searched at the corresponding folder
    showNameList = sorted(os.listdir(config["PATH"]["audio"]))
    showNameList = (
        " ".join(showNameList).replace(config["EXTENSION"]["audio"], "").split()
    )
    # One file mode:
    if input_file_name is not None:
        showNameList = [input_file_name.replace(config["EXTENSION"]["audio"], "")]
        config["PATH"]["audio"] = ""
    # If the output file already exists from a previous call it is deleted
    if os.path.isfile(
        config["PATH"]["output"]
        + config["EXPERIMENT"]["name"]
        + config["EXTENSION"]["output"]
    ):
        os.remove(
            config["PATH"]["output"]
            + config["EXPERIMENT"]["name"]
            + config["EXTENSION"]["output"]
        )
    # Output folder is created
    os.makedirs(config["PATH"]["output"], exist_ok=True)
    # Files are diarized one by one
    for idx, showName in enumerate(showNameList):
        print("\nProcessing file", idx + 1, "/", len(showNameList))
        output_fname = run_diarization(showName, config, sad_file_name)
    return output_fname


def load_rttm(rttm_file_name, min_duration=0.25):
    import pandas as pd

    speaker_track = pd.read_csv(
        rttm_file_name,
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
        speaker_track["duration"] > min_duration
    ]  # keep those longer than min_seconds
    return speaker_track
    # cat_type = CategoricalDtype(categories=speaker_track['speaker_id'].unique().tolist())


def label_waveform_by_speaker(waveform_df, speaker_df):
    waveform_df["ID"] = np.nan
    for i, row in speaker_df.iterrows():
        st = row["start"]
        fn = row["end"]
        waveform_df.loc[
            (waveform_df["time"] > st) & (waveform_df["time"] < fn), "ID"
        ] = row["id"]
    waveform_df["ID"] = waveform_df["ID"].fillna(0).astype("int")
    waveform_df["id_adjusted"] = (waveform_df["ID"] - waveform_df["ID"].min()) / (
        waveform_df["ID"].max() - waveform_df["ID"].min()
    ) - 0.5

    waveform_df["ID"] = waveform_df["ID"].fillna(0).astype("str")

    return waveform_df
