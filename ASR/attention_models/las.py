import json
from copy import deepcopy
from tqdm import tqdm
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import Gumbel
from ASR.asr_api import TorchASRBase

from ASR.asr_api import TorchASRBase
from ASR.utils.general import WORDBANK
from ASR.utils.torch_utils import SpeechSet
from ASR.utils.torch_utils import speech_collate
from ASR.utils.general import convert_numeric_to_transcript

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
        
        #input = input.unsqueeze(1)
        #input = torch.cat((input[:,:,::2,:], input[:,:,1::2,:]), dim=1).mean(axis=1)
        input = input.view(input.shape[0],input.shape[1]//2,2,input.shape[2])
        input = torch.mean(input, 2)
        lens //= 2

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
        self.sm = nn.Softmax(dim=1)

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
        energy = torch.bmm(key, token.unsqueeze(2)).squeeze(2)

        # Create a mask to identify padding values and use that to mask padding values 
        mask = torch.arange(energy.shape[1]).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(device)
        energy.masked_fill_(mask, -1e-9)
        attention = self.sm(energy)

        # Finally multiply with value to return context scaled by probability
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        
        del mask
        return context


class Speller(nn.Module):
    def __init__(self, num_embeds, embedding_dim, hidden_size):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.num_embeds = num_embeds
        self.embedding = nn.Embedding(num_embeds, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(embedding_dim+hidden_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.attention = Attention()
        self.character_prob = nn.Linear(hidden_size*2, num_embeds)

        # Probability p is defined as the teacher forcing probability, and is
        # defined as the probability with which ground truth prediction is shown.
        # This probability is decayed as training progresses 
        self.teacher_forcing_prob = 0.95
        self.gumbel_loc = 0.01

    def forward(self, key, value, device, speech_lens, decode=None):

        # We initialize context with first time-step of encoder's output
        context = key[:,0,:]

        # Initialize hidden and cell states for two LSTMCells
        hidden_0, hidden_1 = torch.zeros(key.shape[0], self.hidden_size).to(device), torch.zeros(key.shape[0], self.hidden_size).to(device)
        cell_0, cell_1 = torch.zeros(key.shape[0], self.hidden_size).to(device), torch.zeros(key.shape[0], self.hidden_size).to(device)

        # Output sequence decodes 
        decodes = []

        prev_pred = torch.zeros(key.shape[0], 1).to(device)

        if decode is not None:
            embedding = self.embedding(decode)
            max_len = decode.shape[1]
        else:
            max_len = 500

        for i in range(max_len):
            # Generate embedding for each character
            # With a probability p, we decide whether the model observes
            # the gold label, or the prediction made at the previous timestep
            if decode is not None:
                if np.random.uniform() < self.teacher_forcing_prob:
                    char_embed = embedding[:,i,:]
                else:
                    prev_pred = Gumbel(prev_pred.to('cpu'), torch.tensor([self.gumbel_loc])).sample().to(device)
                    char_embed = self.embedding(torch.argmax(prev_pred, dim=-1))
            else:
                char_embed = self.embedding(torch.argmax(prev_pred, dim=-1))

            # Concatenate context with character embedding to feed into LSTMCell
            output = torch.cat((char_embed, context), axis=1)
            
            # Pass concatenated context through two LSTM layers
            hidden_0, cell_0 = self.lstm1(output, (hidden_0, cell_0))

            hidden_1, cell_1 = self.lstm2(hidden_0, (hidden_1, cell_1))

            # Obtain context for next time step via attention
            context = self.attention(key, value, hidden_1, speech_lens, device)

            # Obtain a probability distribution over output characters at each time-step
            prev_pred = self.character_prob(torch.cat((hidden_1, context), dim=1))
            decodes.append(prev_pred.unsqueeze(1))

        return torch.cat(decodes, axis=1)



class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                       key_dim, value_dim, num_embeds,
                       embedding_dim, hidden_size,
                       device):
        super(Seq2Seq, self).__init__()
        self.encoder = Listener(input_dim, hidden_dim, key_dim, value_dim)
        self.decoder = Speller(num_embeds, embedding_dim, hidden_size)
        self.device = device

    def forward(self, x, x_len, dec=None):
        key, value, x_len = self.encoder(x, x_len)
        preds = self.decoder(key, value, self.device, x_len, dec)

        return preds



class LAS(TorchASRBase):
    def __init__(self, config):
        super(LAS, self).__init__()
        
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
        self.decode_method = config['models']['LAS']['decode_method']
        
        # Initialize the LAS model
        self.las_model = Seq2Seq(input_dim=input_dim, 
                                 hidden_dim=hidden_size, 
                                 key_dim=key_size, 
                                 value_dim=value_size,
                                 num_embeds=len(WORDBANK),
                                 embedding_dim=embedding_dim,
                                 hidden_size=key_size,
                                 device=self.device
                                )

    def __del__(self):
        del self.las_model   
 
    def create_loader(self, data, transcripts, kwargs):
        '''Helper function for creating Torch dataloader'''
        dataset = SpeechSet(data, transcripts)
        return DataLoader(dataset, **kwargs)

    def train(self, train_data, train_transcripts, 
                    dev_data, dev_transcripts):

        # Define Dataloader arguments for training and validation datasets
        kwargs = {'batch_size' : self.batch_size, 'shuffle' : True,
                'collate_fn' : speech_collate, 
                'num_workers' : 4, "pin_memory":True}

        dev_kwargs = {'batch_size' : 1, 'shuffle' : False,
                    'collate_fn' : speech_collate,
                    'num_workers' : 4, "pin_memory":True}

        # Create training data loader and validation data loader
        trainloader = self.create_loader(train_data, train_transcripts, kwargs)
        devloader = self.create_loader(dev_data, dev_transcripts, dev_kwargs)

        self.las_model = self.las_model.to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(self.las_model.parameters(), self.lr)

        train_tuple = next(iter(trainloader))
        train_sample = (train_tuple[0][0].to(self.device), torch.Tensor(train_tuple[0][1]).long(), train_tuple[1][0].to(self.device))
        #self.writer.add_graph(self.las_model, train_sample)

        # Train and validate model across specified number of epochs
        for epoch in tqdm(range(self.epochs)):

            # Training iteration
            train_decodes, train_groundtruths, train_loss = [], [], []
            self.las_model.train()
            with torch.autograd.set_detect_anomaly(True):
                for batch_num, (speech, decode, truth) in enumerate(trainloader):
                    optimizer.zero_grad()
                    speech_data, speech_lens = speech
                    decode_data, decode_lens = decode
                    truth_data, truth_lens = truth

                    # Create a mask for masking out padding while computing loss
                    #mask = torch.arange(truth_data.shape[1]).unsqueeze(0) < torch.Tensor(truth_lens).long().unsqueeze(1)
                    #mask = mask.to(self.device)

                    # Move data to device
                    speech_data = speech_data.to(self.device)
                    decode_data = decode_data.to(self.device)
                    truth_data = truth_data.to(self.device)
                    speech_lens = torch.Tensor(speech_lens).long()

                    # Obtain probability distribution over targets
                    preds = self.las_model(speech_data, speech_lens, decode_data)
                    #preds_trans = torch.transpose(preds, 1,2).contiguous()
                    preds_trans = preds.contiguous().view(-1, preds.size(-1))

                    mask = torch.zeros(truth_data.size()).to(self.device)
                    for idx, length in enumerate(truth_lens):
                        mask[:length, idx] = 1
                    mask = mask.T.contiguous().view(-1).to(self.device)

                    # Compute loss between generated predictions and ground truth predictions
                    loss = criterion(preds_trans, truth_data.contiguous().view(-1))
                    masked_loss = torch.sum(loss*mask)
                    masked_loss.backward()

                    # Clip gradients and perform forward pass to update parameters
                    nn.utils.clip_grad_norm_(self.las_model.parameters(), 2)
                    optimizer.step()

                    # Collect metrics for evaluation of model's performance
                    train_loss.append(masked_loss.item()/int(torch.sum(mask).item()))
                    if self.decode_method == 'argmax':
                        train_decodes.extend(torch.argmax(preds, dim=2).detach().cpu().numpy().tolist())
                        train_groundtruths.extend(truth_data.detach().cpu().numpy().tolist())

                    if batch_num % 25 == 1:
                        print("Epoch ", epoch, "Training Loss ", train_loss[-1])
                        print(f"Epoch {epoch}/{self.epochs} TRAIN ORIGINAL: ", convert_numeric_to_transcript(train_groundtruths[-1], truncate=True))
                        print(f"Epoch {epoch}/{self.epochs} TRAIN DECODES: ", convert_numeric_to_transcript(train_decodes[-1], truncate=True))
                        print("\n")


                    del speech_data
                    del decode_data 
                    del truth_data
                    del loss
                    del mask
                    del masked_loss
                    del preds
                    del preds_trans
                    torch.cuda.empty_cache()

            avg_train_loss = np.mean(train_loss)
            avg_train_levdist = self.getWER(train_decodes, train_groundtruths)

            self.writer.add_scalar('Loss/Train:', avg_train_loss, epoch)
            self.writer.add_scalar('Levenshtein_Distance/Train:', avg_train_levdist, epoch)
            self.writer.flush()

            # Decay teacher forcing rate every 4 epochs
            if epoch % self.teacher_forcing_decay == 0 and self.teacher_forcing_decay >=0.55:
                self.las_model.decoder.teacher_forcing_prob -= 0.05
            
            # Validate the model on dev data
            self.las_model.eval()
            dev_decodes, dev_groundtruths = [], []
            min_lev_dist = float("inf")
            for batch_num, (speech, decode, truth) in enumerate(devloader):
                speech_data, speech_lens = speech
                decode_data, decode_lens = decode
                truth_data, truth_lens = truth

                # Create a mask for masking out padding while computing loss
                mask = torch.arange(truth_data.shape[1]).unsqueeze(0) < torch.Tensor(truth_lens).long().unsqueeze(1)
                mask = mask.to(self.device)

                # Move data to device
                speech_data = speech_data.to(self.device)
                decode_data = decode_data.to(self.device)
                truth_data = truth_data.to(self.device)
                speech_lens = torch.Tensor(speech_lens).long()

                # Obtain probability distribution over targets
                preds = self.las_model(speech_data, speech_lens)
                preds_trans = torch.transpose(preds, 1,2).contiguous()

                # Collect metrics for evaluation of model's performance
                if self.decode_method == 'argmax':
                    dev_decodes.extend(torch.argmax(preds, dim=2).detach().cpu().numpy().tolist())
                    dev_groundtruths.extend(truth_data.detach().cpu().numpy().tolist())

                if batch_num % 25 == 1:
                    print(f"Epoch {epoch}/{self.epochs} DEV ORIGINAL: ", convert_numeric_to_transcript(dev_groundtruths[-1], truncate=True))
                    print(f"Epoch {epoch}/{self.epochs} DEV DECODES: ", convert_numeric_to_transcript(dev_decodes[-1], truncate=True))
                    print("\n")

                del speech_data
                del decode_data 
                del truth_data
                del preds
                del preds_trans
                torch.cuda.empty_cache()

            avg_dev_levdist = self.getWER(dev_decodes, dev_groundtruths)

            # Copies the model state for the best perfoming epoch over validation set
            if avg_dev_levdist < min_lev_dist:
                self.dics = deepcopy(self.las_model.state_dict())
                min_lev_dist = avg_dev_levdist
                self.saveToFile(dev_decodes, dev_groundtruths, 'decodes/LAS.txt')

            self.writer.add_scalar('Levenshtein_Distance/Dev:', avg_dev_levdist, epoch)
            self.writer.flush()

        
        # Loads model with best performance on validation set
        self.las_model.load_state_dict(self.dics)

    
    def evaluate(self, test_data, test_transcripts):
        test_kwargs = {'batch_size' : self.batch_size, 'shuffle' : False,
                    'collate_fn' : speech_collate,
                    'num_workers' : 4, 'pin_memory':True}        

        self.testloader = self.create_loader(test_data, test_transcripts, test_kwargs)

        # Validate the model on dev data
        self.las_model.eval()
        test_decodes, test_groundtruths = [], []
        for batch_num, (speech, decode, truth) in enumerate(self.testloader):
            speech_data, speech_lens = speech
            decode_data, decode_lens = decode
            truth_data, truth_lens = truth

            # Move data to device
            speech_data = speech_data.to(self.device)
            decode_data = decode_data.to(self.device)
            truth_data = truth_data.to(self.device)
            speech_lens = torch.Tensor(speech_lens).long()

            # Obtain probability distribution over targets
            preds = self.las_model(speech_data, speech_lens)

            # Collect metrics for evaluation of model's performance
            if self.decode_method == 'argmax':
                test_decodes.extend(torch.argmax(preds, dim=2).detach().cpu().numpy().tolist())
                test_groundtruths.extend(truth_data.detach().cpu().numpy().tolist())


            del speech_data
            del decode_data 
            del truth_data
            del preds
            torch.cuda.empty_cache()

        avg_test_levdist = self.getWER(test_decodes, test_groundtruths)

        self.writer.add_scalar('Levenshtein_Distance/Test:', avg_test_levdist)
        self.writer.flush()