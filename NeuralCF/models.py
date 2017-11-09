import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from spotlight.layers import ScaledEmbedding, ZeroEmbedding
from spotlight.losses import regression_loss
from spotlight.torch_utils import minibatch, set_seed, shuffle

from layers import NeuralCF, BilinearNet


class ExplicitNeuralCFModel(object):
    """
    [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
    """

    def __init__(self,
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 random_state=None):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._random_state = random_state or np.random.RandomState()

        self._num_users = None
        self._num_items = None
        self._net = None
        self._loss_func = None

        set_seed(self._random_state.randint(-10**8, 10**8))

    # def __repr__(self):
    #     return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        # self._net = BilinearNet(self._num_users,
        #                     self._num_items,
        #                     self._embedding_dim)
        self._net = NeuralCF(self._num_users,
                            self._num_items,
                            self._embedding_dim)
        
        self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate)

        self._loss_func = regression_loss

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions, verbose=True):
        """
        Fit the model.
        """

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(user_ids, item_ids)

        for epoch_num in range(self._n_iter):

            users, items, ratings = shuffle(user_ids,
                                            item_ids,
                                            interactions.ratings,
                                            random_state=self._random_state)

            user_ids_tensor = torch.from_numpy(users)
            item_ids_tensor = torch.from_numpy(items)
            ratings_tensor  = torch.from_numpy(ratings)

            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_user,
                  batch_item,
                  batch_ratings)) in enumerate(minibatch(user_ids_tensor,
                                                         item_ids_tensor,
                                                         ratings_tensor,
                                                         batch_size=self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                ratings_var = Variable(batch_ratings)

                predictions = self._net(user_var, item_var)

                self._optimizer.zero_grad()

                loss = self._loss_func(ratings_var, predictions)
                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._net.train(False)

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_ids = Variable(user_ids).squeeze()
        item_ids = Variable(item_ids).squeeze()
        
        out = self._net(user_ids, item_ids)
        return out.data.numpy().flatten()