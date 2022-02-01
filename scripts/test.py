import sys
sys.path.append('../')

from ASR.main import run_model
from ASR.attention_models.las import LAS

if __name__=="__main__":

    run_model(LAS, '/zfsauton/data/public/vsanil/speech/LibriSpeech')