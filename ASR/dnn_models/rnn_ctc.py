import json
from copy import deepcopy
from tqdm import tqdm
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from ASR.main import ASRBase
from ASR.utils.general import WORDBANK
from ASR.utils.torch_utils import SpeechSet
from ASR.utils.torch_utils import speech_collate


class RNNCTC(ASRBase):
    def __init__(self, config):
        self.trainloader, self.devloader = None, None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set global parameters
        input_dim = config['mfcc_bands']

        self.dics = None    # Stores model weights

        