import json
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ASR.utils.general import WORDBANK
from ASR.utils.torch_utils import SpeechSet
from ASR.utils.torch_utils import speech_collate

from torch.utils.tensorboard import SummaryWriter

class Listener(nn.Module):
    def __init__(self, input_size, hidden_size, key_dim=128, value_dim=128):
        super(Listener, self).__init__()
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

        return key, value, x_len

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, key, value, token, lens, device):
        '''
            Returns attention by conducting batch matrix multiplication between the Listener's speech 
            representations, and the decoder's LSTMCell's output at the current time step to generate
            context for the next time step.

            Arguments:
                Key: (Batch Size, Sequence Length, Num features)
                Value: (Batch Size, Sequence Length, Num features)

        '''
        # Obtain energy in the form of a probability distribution by batch-multiplying decoder output with encoder representation
        energy = torch.bmm(key, token.unsqueeze(2))
        energy = torch.softmax(energy, axis=1)

        # Create a mask to identify padding values and use that to mask padding values 
        mask = torch.arange(key.shape[1]).unsqueeze(0) >= lens.unsqueeze(1)
        energy.masked_fill_(mask.unsqueeze(2).to(device), 1e-9)

        # Finally multiply with value to return context scaled by probability
        context = torch.bmm(torch.transpose(value, 1,2), energy)[:,:,0]
        
        return context


class Speller(nn.Module):
    def __init__(self, num_embeds, embedding_dim, hidden_size):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeds, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(embedding_dim+hidden_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.attention = Attention()
        self.fc_layer = nn.Linear(hidden_size, num_embeds)

    def forward(self, key, value, decode, decode_lens, device):

        # We initialize context with first time-step of encoder's output
        context = key[:,0,:]

        # Initialize hidden and cell states for two LSTMCells
        hidden_0, hidden_1 = torch.zeros(key.shape[0], self.hidden_size).to(device), torch.zeros(key.shape[0], self.hidden_size).to(device)
        cell_0, cell_1 = torch.zeros(key.shape[0], self.hidden_size).to(device), torch.zeros(key.shape[0], self.hidden_size).to(device)

        # Output sequence decodes 
        decodes = []

        for i in range(decode.shape[0]):
            # Generate embedding for each character
            char_embed = self.embedding(decode[:, i])

            # Concatenate context with character embedding to feed into LSTMCell
            output = torch.cat((context, char_embed), axis=1)
            
            # Pass concatenated context through two LSTM layers
            hidden_0, cell_0 = self.lstm1(output, (hidden_0, cell_0))

            hidden_1, cell_1 = self.lstm2(hidden_0, (hidden_1, cell_1))

            # Obtain a probability distribution over output characters at each time-step
            out_probs = self.fc_layer(hidden_1)
            decodes.append(out_probs)

            # Obtain context for next time step via attention
            context = self.attention(key, value, hidden_1, decode_lens, device)

        return torch.cat(decodes, axis=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                       key_dim, value_dim, num_embeds,
                       embedding_dim, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Listener(input_dim, hidden_dim, key_dim, value_dim)
        self.decoder = Speller(num_embeds, embedding_dim, hidden_size)

    def forward(self, x, x_len, dec, dec_len, device):
        key, value, x_len = self.encoder(x, x_len)
        preds = self.decoder(key, value, dec, x_len, device)

        return preds

class LAS(object):
    # TODO : Convert this class to inherit from a base ASR class
    def __init__(self, config):
        self.model = None
        self.trainloader, self.devloader = None, None
        
        # Set global parameters
        input_dim = config["mfcc_bands"]

        # Set model specific parameters
        self.lr = config['models']['LAS']['lr']
        self.epochs = config['models']['LAS']['epochs']
        self.batch_size = config['models']['LAS']['batch_size']
        hidden_size = config['models']['LAS']['hidden_size']
        key_size = config['models']['LAS']['key_size']
        value_size = config['models']['LAS']['value_size']
        embedding_dim = config['models']['LAS']['embedding_dim']
        self.teacher_forcing_decay = config['models']['LAS']['teacher_forcing_decay']
        
        # Initialize the LAS model
        self.las_model = Seq2Seq(input_dim=input_dim, 
                                 hidden_dim=hidden_size, 
                                 key_dim=key_size, 
                                 value_dim=value_size,
                                 num_embeds=len(WORDBANK),
                                 embedding_dim=embedding_dim,
                                 hidden_size=key_size
                                )
                                 
 
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

        # Create a writer object for logging training to Tensorboard
        #writer = SummaryWriter()

        # Train and validate model across specified number of epochs
        for epoch in range(self.epochs):

            # Training iteration
            for batch_num, (speech, decode, truth) in enumerate(trainloader):
                speech_data, speech_lens = speech
                decode_data, decode_lens = decode
                truth_data, truth_lens = truth

                # Move data to device
                speech_data = speech_data.to(device)
                decode_data = decode_data.to(device)

                # Obtain probability distribution over targets
                preds = self.las_model(speech_data, speech_lens, decode_data, decode_lens, device=device)

                print(speech_data.shape)
                print(preds[0].shape, preds[1].shape)

                #writer.close()
                raise KeyboardInterrupt
