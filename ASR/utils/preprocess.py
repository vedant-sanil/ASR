import os
import json
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
from librosa.feature import mfcc

def get_data_transcripts(path, sampling_rate=16000, mfcc_bands=40):
    featurized_data, transcripts = [], []
    for root, dirs, files in tqdm(os.walk(path, topdown=False)):
        if len(files) > 0:
            trans_filename = '-'.join(root.split('/')[-2:])+'.trans.txt'
            transcript_file = os.path.join(root, trans_filename)
            with open(transcript_file, 'r') as tf:
                for lines in tf:
                    filename, transcript = lines.split(" ", 1)
                    filename += '.flac'
                    filepath = os.path.join(root, filename)

                    data, samplerate = sf.read(filepath, dtype='float32')
                    if samplerate != sampling_rate:
                        data = librosa.resample(data, samplerate, sampling_rate)
                        samplerate = sampling_rate

                    mfcc_feats = mfcc(data, samplerate, n_mfcc=mfcc_bands).T

                    featurized_data.append(mfcc_feats)
                    transcripts.append(transcript)

    return np.array(featurized_data, dtype='object'), np.array(transcripts, dtype='object')


def preprocess(feature_path, libripath, config_dict):
    # Create data paths
    train_path = os.path.join(libripath, 'train-clean-100')
    dev_path = os.path.join(libripath, 'dev-clean')
    test_path = os.path.join(libripath, 'test-clean')

    train_data, train_transcripts = get_data_transcripts(train_path, config_dict['sampling_rate'], config_dict['mfcc_bands'])
    dev_data, dev_transcripts = get_data_transcripts(dev_path, config_dict['sampling_rate'], config_dict['mfcc_bands'])
    test_data, test_transcripts = get_data_transcripts(test_path, config_dict['sampling_rate'], config_dict['mfcc_bands'])

    # Save each file as a numpy array
    with open(f'{feature_path}/train/train-clean-100.npy', 'wb') as dtfile:
        np.save(dtfile, train_data)

    with open(f'{feature_path}/train/train-clean-100-transcripts.npy', 'wb') as dtfile:
        np.save(dtfile, train_transcripts)

    with open(f'{feature_path}/dev/dev-clean.npy', 'wb') as dtfile:
        np.save(dtfile, dev_data)

    with open(f'{feature_path}/dev/dev-clean-transcripts.npy', 'wb') as dtfile:
        np.save(dtfile, dev_transcripts)

    with open(f'{feature_path}/test/test-clean.npy', 'wb') as dtfile:
        np.save(dtfile, test_data)

    with open(f'{feature_path}/test/test-clean-transcripts.npy', 'wb') as dtfile:
        np.save(dtfile, test_transcripts)