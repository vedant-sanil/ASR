import json
from copy import deepcopy
from tqdm import tqdm
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ASR.asr_api import TorchASRBase
from ASR.utils.general import WORDBANK


class RNN_CTC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 num_layers):
        super(RNN_CTC, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

    def forward(self, X, lengths):
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)


class RNNCTC(TorchASRBase):
    def __init__(self, config):
        super(RNNCTC, self).__init__()

        input_dim = config["mfcc_bands"]

        # Set model specific parameters
        self.lr = config['models']['RNN-CTC']['lr']
        self.epochs = config['models']['RNN-CTC']['epochs']
        self.batch_size = config['model']['RNN-CTC']['batch_size']

        # Initialize the RNN-CTC model

        