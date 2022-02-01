import json
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ASR.utils.torch_utils import SpeechSet
from ASR.utils.torch_utils import speech_collate

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, key_dim=128, value_dim=128):
        super(Encoder, self).__init__()
        self.init_layer = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.pblstm1 = self._make_layer(hidden_size*2, hidden_size)
        self.pblstm2 = self._make_layer(hidden_size*2, hidden_size)
        self.pblstm3 = self._make_layer(hidden_size*2, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.key_layer = nn.Linear(hidden_size, key_dim)
        self.value_layer = nn.Linear(hidden_size, value_dim)
                
    def _make_layer(self, input_size, hidden_size):
        '''To be used if advanced dropout methods are to be added to LSTMs'''

        return nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, num_layers=1)

    def _reshape_and_pool(self, input):
        '''Reshapes and pools tensor such that the length is halved'''
        input, lens = nn.utils.rnn.pad_packed_sequence(input, batch_first=True)
        if input.shape[1] % 2 == 1:
            input = input[:,:-1,:]
            lens -= 1
        
        input = input[:, None, :, :]
        input = torch.cat((input[:,:,::2,:], input[:,:,1::2,:]), dim=1).mean(axis=1)
        lens /= 2

        input = nn.utils.rnn.pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False)
        return input, lens


    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x = self.init_layer(x)[0]

        # Pass through three pyramidal BLSTM layers
        x, x_len = self._reshape_and_pool(x)
        x = self.pblstm1(x)[0]

        x, x_len = self._reshape_and_pool(x)
        x = self.pblstm2(x)[0]

        x, x_len = self._reshape_and_pool(x)
        x = self.pblstm3(x)[0]

        x, x_len = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Pass network through any FC layers before splitting into two outputs: key and value
        # This split is useful for an intuitive understanding of how attention works
        x = self.fc1(x)
        key, value = self.key_layer(x), self.value_layer(x)

        return key, value

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)

    def forward(self, x, x_len, dec, dec_len):
        x = self.encoder(x, x_len)

        return x

class LAS(object):
    def __init__(self, config):
        self.model = None
        self.trainloader, self.devloader = None, None
        
        # Set global parameters
        self.input_dim = config["mfcc_bands"]

        # Set model specific parameters
        self.lr = config['models']['LAS']['lr']
        self.epochs = config['models']['LAS']['epochs']
        self.batch_size = config['models']['LAS']['batch_size']
        self.hidden_size = config['models']['LAS']['hidden_size']
        self.teacher_forcing_decay = config['models']['LAS']['teacher_forcing_decay']
        
        # Initialize the LAS model
        self.las_model = Seq2Seq(self.input_dim, self.hidden_size)
 
    def _create_loader(self, data, transcripts, kwargs):
        '''Helper function for creating Torch dataloader'''
        dataset = SpeechSet(data, transcripts)
        return DataLoader(dataset, **kwargs)

    def train(self, train_data, train_transcripts, 
                    dev_data, dev_transcripts):

        # Define Dataloader arguments for training and validation datasets
        kwargs = {'batch_size' : self.batch_size, 'shuffle' : True,
                'collate_fn' : speech_collate, 
                'num_workers' : 0}

        dev_kwargs = {'batch_size' : self.batch_size, 'shuffle' : False,
                    'collate_fn' : speech_collate,
                    'num_workers' : 1}

        # Create training data loader and validation data loader
        trainloader = self._create_loader(train_data, train_transcripts, kwargs)
        devloader = self._create_loader(dev_data, dev_transcripts, dev_kwargs)

        # Define model specific 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.las_model = self.las_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.las_model.parameters(), self.lr)

        # Train and validate model across specified number of epochs
        for i in range(self.epochs):

            # Training iteration
            for batch_num, (speech, decode, truth) in enumerate(trainloader):
                speech_data, speech_lens = speech
                decode_data, decode_lens = decode
                truth_data, truth_lens = truth

                # Move data to device
                speech_data = speech_data.to(device)

                # Obtain probability distribution over targets
                preds = self.las_model(speech_data, speech_lens, decode_data, decode_lens)

                print(speech_data.shape)
                print(preds[0].shape, preds[1].shape)

                raise KeyboardInterrupt
