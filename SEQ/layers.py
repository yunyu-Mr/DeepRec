import torch
import torch.nn as nn
import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class SeqCNN(nn.Module):
    """
    Causal CNN for sequences.
    """
    def __init__(self, num_items, embedding_dim=32, num_layers=1, activate='tanh', sparse=False):

        super(SeqCNN, self).__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               padding_idx=0,
                                               sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=0)
        
        # Convolution layers
        self.kernel_size = k = 3
        self.num_layers = num_layers
        self.convs = [
            nn.Conv1d(embedding_dim,  # Cin
                      embedding_dim,  # Cout
                      kernel_size=k) 
            for _ in range(num_layers)
        ]
        
        # Activation function
        if activate == 'relu':
            activates = [F.relu for _ in range(num_layers)]
        elif activate == 'tanh':
            activates = [F.tanh for _ in range(num_layers)]
        else:
            activates = [F.relu for _ in range(num_layers-1)] + [F.tanh]
        self.activates = activates
        
    def user_representation(self, sequences):
        emb_seq = self.item_embeddings(sequences)  # (N, L, E)
        emb_seq = emb_seq.permute(0, 2, 1)         # (N, E, L), embedding_dim is the channels
        
        x = F.pad(emb_seq,                  # (N, E, k + L)
                  (self.kernel_size, 0))
        x = self.activates[0](self.convs[0](x)) # (N, E, 1 + L)
        
        # Residual
        x = x + F.pad(emb_seq, (1, 0))      # (N, E, 1 + L)
        
        # Rest layers
        for i in range(1, self.num_layers):
            residual = x
            x = F.pad(x, (self.kernel_size - 1, 0))
            x = self.activates[i](self.convs[i](x))
            x = x + residual
        
        return x[:,:,:-1], x[:,:,-1:]       # (N, E, L),  (N, E, 1)
        
    def forward(self, user_representation, targets):
        """
        user_representation: (N, E, L)
        targets: (N, L)
        """
        emb_target = self.item_embeddings(targets).permute(0,2,1) # (N, E, L)
        
        b_i = self.item_biases(targets).squeeze()  # (N, L)
        
        dot = (user_representation * emb_target).sum(1).squeeze() # (N, L)
        return dot + b_i
