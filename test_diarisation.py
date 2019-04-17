import torch
from pyannote.audio.labeling.extraction import SequenceLabeling
import logging as lg
import os
from src.sad_diarisation_bk import generate_sad_lbl, run_pyBK_diarisation

if __name__ == '__main__':
    SAD_MODEL = ('models/speech_activity_detection/train/'
                 'AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt')
    sad = SequenceLabeling(model=SAD_MODEL, device=torch.device('cuda'))

    input_filename = f'data/raw/carell.wav'
    # rttm data/processed/carell_galifianakis/carell_galifianakis.rttm
    basename = os.path.splitext(os.path.basename(input_filename))[0]
    sad_folder = f'data/sad/'
    config_loc ='./models/config.ini'
    diarisation_output_folder = f'data/processed/{basename}/'
    lg.info('generating SADs...')
    sad_filename = generate_sad_lbl(input_filename, sad_folder, model=sad)
    lg.info('doing diarisation...')
    output_name = run_pyBK_diarisation(config_loc=config_loc,input_file_name=input_filename,output_folder=diarisation_output_folder,output_name=basename,sad_file_name = sad_filename)
    print(output_name)
    lg.info('Done with speaker identification..')