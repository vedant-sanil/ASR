import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from Levenshtein import distance

from torch.utils.tensorboard import SummaryWriter

from ASR.utils.torch_utils import SpeechSet
from ASR.utils.general import convert_numeric_to_transcript

class ASRBase(object):
    def __init__(self):
        raise NotImplementedError

    def create_loader(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def getWER(self, decode, groundtruth):
        '''
            Computes average levenshtein distance over a list 
            of decoded and ground truth strings.
        '''
        lev_distances = []
        for i in range(len(decode)):
            lev_distances.append(distance(convert_numeric_to_transcript(decode[i], truncate=True), 
                                          convert_numeric_to_transcript(groundtruth[i], truncate=True)))

        return np.mean(lev_distances)

    def saveToFile(self, decode, groundtruth, filepath):
        '''
            Saves decodings to a file
        '''
        with open(filepath, 'w') as f:
            for i in range(len(decode)):
                f.write(f'ACTUAL: {convert_numeric_to_transcript(groundtruth[i], truncate=True)}\n')
                f.write(f'DECODED: {convert_numeric_to_transcript(decode[i], truncate=True)}\n')
                f.write('\n')



class TorchASRBase(ASRBase):
    def __init__(self):
        self.trainloader, self.devloader, self.testloader = None, None, None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dics = None        # Stores best model's weights

        # Create a writer object for logging training to Tensorboard
        self.writer = SummaryWriter(flush_secs=10, filename_suffix='test')

    def create_loader(self, data, transcripts, kwargs):
        '''Helper function for creating a Torch dataloader'''
        dataset = SpeechSet(data, transcripts)
        return DataLoader(dataset, **kwargs)

    def train():
        raise NotImplementedError

    def evaluate():
        raise NotImplementedError