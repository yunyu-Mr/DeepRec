import numpy as np


def sliding_windows(sequence, window_size=200, step_size=200):
    """
    Sliding window sub-sequences.
    """
    for i in range(len(sequence), 0, -step_size):
        yield sequence[max(0, i - window_size) : i]

def pad_sequences(sequences, maxlen=200):
    """
    Pad zeros to the left of sequence.
    """
    padded = np.zeros((len(sequences), maxlen), 
                      dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            seq = seq[-maxlen:]
        padded[i, -len(seq):] = seq
    
    return padded