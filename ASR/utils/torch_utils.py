import torch
import torch.nn as nn
from torch.utils.data import Dataset

import ASR.utils.general as general

class SpeechSet(Dataset):
    def __init__(self, data, transcripts, return_single_transcript=False):
        self.data = data
        self.transcripts = transcripts
        self.return_single_transcript = return_single_transcript

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        speech, transcript = self.data[index], self.transcripts[index]

        # Append start of sentence and end of sentence tokens and convert
        # to numeric tensor
        transcript = torch.from_numpy(general.convert_trancript_to_numeric(transcript)).long()

        # Return speech and transcripts as torch tensors and split transcripts
        # for decoding and ground truth

        return torch.from_numpy(speech).float(), transcript[:-1], transcript[1:]


def speech_collate(batch):
    '''Pads each batch using 0 padded value and returns padded value with lengths'''
    speech, decode, ground_truth = zip(*batch)
    batch_size = len(speech)

    # Calculate length of each batch element
    speech_lens = [len(speech[i]) for i in range(batch_size)]
    decode_lens = [len(decode[i]) for i in range(batch_size)]
    ground_truth_lens = [len(ground_truth[i]) for i in range(batch_size)]

    # Pad each sequence 
    speech = nn.utils.rnn.pad_sequence(speech, batch_first=True)
    decode = nn.utils.rnn.pad_sequence(decode, batch_first=True)
    ground_truth = nn.utils.rnn.pad_sequence(ground_truth, batch_first=True)

    return (speech, speech_lens), (decode, decode_lens), (ground_truth, ground_truth_lens)
