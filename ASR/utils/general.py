from typing import List
import numpy as np

WORDBANK = {"@" : 0, 
            "<eos>" : 1,
            "<sos>" : 2, 
            "A" : 3, 
            "B" : 4, 
            "C" : 5, 
            "D" : 6, 
            "E" : 7, 
            "F" : 8, 
            "G" : 9, 
            "H" : 10,
            "I" : 11, 
            "J" : 12, 
            "K" : 13, 
            "L" : 14, 
            "M" : 15, 
            "N" : 16, 
            "O" : 17, 
            "P" : 18, 
            "Q" : 19, 
            "R" : 20, 
            "S" : 21, 
            "T" : 22,
            "U" : 23, 
            "V" : 24, 
            "W" : 25, 
            "X" : 26, 
            "Y" : 27, 
            "Z" : 28, 
            " " : 29, 
            "'" : 30, 
            "." : 31, 
            "," : 32, 
            "!" : 33, 
            "?" : 34,
            "-" : 35}

INDEXMAP = {v:k for k,v in WORDBANK.items()}


def convert_trancript_to_numeric(transcript):
    '''Converts a transcript string into its numeric embedding and returns it as pytorch string'''

    word_ls = [WORDBANK['<sos>']] + [WORDBANK[t] for t in transcript.strip()] + [WORDBANK['<eos>']]
    return np.array(word_ls)


def convert_numeric_to_transcript(numeric : List, truncate=False):
    '''Converts an array of numeric indices into their corresponding characters'''
    
    if truncate:
        ls = []
        for n in numeric:
            if n == 1:
                break
            ls.append(INDEXMAP[n])
            
        return ''.join(ls)
    else:
        return ''.join([INDEXMAP[n] for n in numeric])