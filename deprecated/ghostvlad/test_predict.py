#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:56:16 2019

@author: danielsola
"""

import model

unique_list = "z-c-feisty.wav"

params = {
    "dim": (257, None, 1),
    "nfft": 512,
    "min_slice": 720,
    "win_length": 400,
    "hop_length": 160,
    "n_classes": 5994,
    "sampling_rate": 16000,
    "normalize": True,
}

network_eval = model.vggvox_resnet2d_icassp(
    input_dim=params["dim"], num_class=params["n_classes"], mode="eval"
)
