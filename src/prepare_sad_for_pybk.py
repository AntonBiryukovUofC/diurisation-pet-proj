from pyannote.audio.labeling.extraction import SequenceLabeling
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyannote.audio.signal import Binarize
import os


def generate_sad_lbl(filename_input = f'../data/raw/z-c-feisty.wav',output_folder =f'../data/sad/',onset = 0.7,offset = 0.7,model = None):
    basename = os.path.splitext(os.path.basename(filename_input))[0]
    output_filename = os.path.join(output_folder,f'{basename}.lbl')
    test_file = {'uri': 'filename', 'audio': filename_input}
    sad_scores = model(test_file)
    # binarize raw SAD scores (as `pyannote.core.Timeline` instance)
    # NOTE: both onset/offset values were tuned on AMI dataset.
    # you might need to use different values for better results.
    binarize = Binarize(offset=offset, onset=onset, log_scale=True)
    speech = binarize.apply(sad_scores, dimension=1)
    # iterate over speech segments (as `pyannote.core.Segment` instances)
    print(f'Length of speech segment container = {len(speech)}')
    fstr = [f'{i.start} {i.end} speech' for i in speech]
    sad_content = '\n'.join(fstr)
    with open(output_filename, 'w') as f:
        f.write(sad_content)


if __name__ == '__main__':
    SAD_MODEL = ('../models/speech_activity_detection/train/'
                 'AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt')
    sad = SequenceLabeling(model=SAD_MODEL, device=torch.device('cuda'))

    generate_sad_lbl(f'../data/raw/z-c-feisty.wav',f'../data/sad/',model = sad)
