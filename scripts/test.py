import os
import sys
import argparse
sys.path.append('../')

from ASR.main import run_model
from ASR.attention_models.las import LAS

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu", default="1", help="GPU to use")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    run_model(LAS, '/zfsauton/data/public/vsanil/speech/LibriSpeech')