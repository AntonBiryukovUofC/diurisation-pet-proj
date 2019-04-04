import torch
from pyannote.audio.labeling.extraction import SequenceLabeling

from src.sad_diarisation_bk import generate_sad_lbl, run_pyBK_diarisation

if __name__ == '__main__':
    SAD_MODEL = ('models/speech_activity_detection/train/'
                 'AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt')
    sad = SequenceLabeling(model=SAD_MODEL, device=torch.device('cuda'))

    generate_sad_lbl(f'data/raw/z-c-feisty.wav', f'data/sad/', model=sad)
    run_pyBK_diarisation(config_loc='./models/config.ini')
