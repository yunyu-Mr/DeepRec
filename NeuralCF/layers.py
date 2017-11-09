import torch.nn as nn
from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class NeuralCF(nn.Module):
    """
    NeuralCF representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by neural network (MLP).
    """

    def __init__(self, num_users, num_items, embedding_dim=32):

        super(NeuralCF, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)
                
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.
        """

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        
        pointwise = user_embedding * item_embedding
        linear = self.linear(pointwise).squeeze()
        
        return linear + user_bias + item_bias


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.
    """

    def __init__(self, num_users, num_items, embedding_dim=32):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.
        """

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias