import os
import json
import logging
import numpy as np

from ASR.utils.preprocess import preprocess

def train(model, train_data, train_transcripts, dev_data=None, dev_transcripts=None):


    model.train(train_data, train_transcripts, dev_data, dev_transcripts)

    return model

def evaluate(model, test_data, test_transcripts):
    model.evaluate(test_data, test_transcripts)

    return model

def load_data(feature_path):
    r""" Loads the pre-processed data saved as numpy objects. Data is 
    saved as n-band MFCC feature vectors and their corresponding 
    transcripts.
    """

    train_path, dev_path, test_path = os.path.join(feature_path, 'train'), \
                                      os.path.join(feature_path, 'dev'), \
                                      os.path.join(feature_path, 'test')

    with open(os.path.join(train_path, 'train-clean-100.npy'), 'rb') as tfile:
        train_data = np.load(tfile, allow_pickle=True)

    with open(os.path.join(train_path, 'train-clean-100-transcripts.npy'), 'rb') as tfile:
        train_data_transcripts = np.load(tfile, allow_pickle=True)

    with open(os.path.join(dev_path, 'dev-clean.npy'), 'rb') as tfile:
        dev_data = np.load(tfile, allow_pickle=True)

    with open(os.path.join(dev_path, 'dev-clean-transcripts.npy'), 'rb') as tfile:
        dev_data_transcripts = np.load(tfile, allow_pickle=True)

    with open(os.path.join(test_path, 'test-clean.npy'), 'rb') as tfile:
        test_data = np.load(tfile, allow_pickle=True)

    with open(os.path.join(test_path, 'test-clean-transcripts.npy'), 'rb') as tfile:
        test_data_transcripts = np.load(tfile, allow_pickle=True)

    return train_data, train_data_transcripts, dev_data, dev_data_transcripts, test_data, test_data_transcripts


def run_model(model, libripath):
    r""" The main function used for running the model. Data will be pre-processed
    if it hasn't already, and the model is trained and evaluated.
    """

    config_dict = {}
    with open('../config.json', 'r') as cfg:
        config_dict = json.load(cfg)
    
    model = model(config_dict)
    
    feature_path = os.path.join(libripath, 'features')
    if len(os.listdir(os.path.join(feature_path, 'train'))) == 0:
        print("Pre-processing data")
        preprocess(feature_path, libripath, config_dict)
    
    print("Loading pre-processed data")
    train_data, train_transcripts, dev_data, dev_transcripts, test_data, test_transcripts = load_data(feature_path)

    model = train(model, train_data, train_transcripts, dev_data, dev_transcripts)
    
