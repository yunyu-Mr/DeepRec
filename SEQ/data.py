import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder
from sequence_utils import sliding_windows, pad_sequences


class Dataset():
    """
    Dataset store user-item interaction records in pd.DataFrame

    Make train test split, and train_plus as well.
    """
    def __init__(self, csvfile=None, df=None, 
                 sample=1.0, 
                 cut_item=100,
                 num_test_users=500, 
                 random_state=None):
        # Random state for stable test
        if random_state is None:
            random_state = np.random.RandomState(123)
        self.random_state = random_state
        
        if csvfile is not None:
            df = self._load_data(csvfile, sample=sample, cut_item=cut_item)
        elif df is None:
            raise RuntimeError("No input file or DataFrame.")
        
        self.num_test_users = num_test_users
        self.users = df['user'].unique()
        self.items = df['item'].unique()

        self.n_user = self.users.size
        self.n_item = self.items.size
        print("#User: {}\t#Item: {}".format(self.n_user, self.n_item))

        self.max_item = self.items.max()

        self._train_test_split(df)

    def _load_data(self, interaction_file, sample=1.0, cut_item=100):
        df = pd.read_csv(interaction_file)
        df.columns = ['user', 'item']
        
        # Remove non-pop items.
        cnt = df.item.value_counts()
        rare_items = cnt[cnt < cut_item].index
        df = df.loc[~ df.item.isin(rare_items)]
        
        # Sampling
        users = df['user'].unique()
        users = self.random_state.choice(users, replace=False,
                                         size = math.ceil(sample*len(users)))
        df = df[df.user.isin(users)]

        # One-hot encoding, label from 1 to n
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]) + 1  # Zero for padding, so remove it.
        return df

    def _train_test_split(self, df):
        # Train test split
        test_users = self.random_state.choice(self.users, size=self.num_test_users)  # Random select some users for test

        self.train = train = df[~df.user.isin(test_users)]
        train_items = train['item'].unique()  # All items in trainset

        test = df[df.user.isin(test_users)]
        test = test[test['item'].isin(train_items)]  # Remove items that not in trainset

        print(len(train), len(test))

        # Get head 80% interactions for each user.
        TOP_FRAC = 0.8
        def _get_top(x, frac=0.8):
            n_top = math.ceil(len(x) * frac)
            return x.head(n_top)
        test_first = test.groupby('user').apply(_get_top, frac=TOP_FRAC).reset_index(drop=True)

        def _get_tail(x, frac=0.2):
            n_tail = math.floor(len(x)*frac)
            return x.tail(n_tail)
        self.test_second = test_second = test.groupby('user').apply(_get_tail, frac=1-TOP_FRAC).reset_index(drop=True)

        # Extended trainset, including the first part of testset.
        self.train_plus = pd.concat([train, test_first])
        self.test = test_first

    def _gen_sequences(self, df):
        sequences = df.groupby('user').agg({'item': lambda x:list(x)})
        sequences.columns = ['item_sequence']
        return sequences.reset_index()

    def get_train_test_sequences(self):
        # Gen sequences for train and test
        train = self._gen_sequences(self.train)
        test  = self._gen_sequences(self.test)

        # Remaining part for evaluation.
        test_second = self._gen_sequences(self.test_second)
        test['eval_sequence'] = test_second['item_sequence']

        return train, test
    
    
class MiniBatcher():
    """
    Create mini batch for df (dataframe of sequences).
    
    First sliding windows (long sequences would be cut into several pices). Then, zero padding to the left.
    
    While iterate the dataset, sequences would be shuffled. And negative items would be generated.
    """
    def __init__(self, df, n_items, 
                 batch_size=128, 
                 maxlen=200,
                 minlen=20,
                 shuffle=True,
                 sampling_prob=None,
                 random_state=None):
        
        self._n_items = n_items
        self.maxlen = maxlen
        self.minlen = minlen
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampling_prob = sampling_prob
        
        if random_state is None:
            random_state = np.random.RandomState(123)
        self.random_state = random_state
        
        # Gen sequence
        sequences = self._gen_sequences(df)
        user_ids, sequences = zip(*sequences)   # It's memory-consuming, but faster to gen mini batch.
        self.user_ids = user_ids
        # Pad zeros
        self.sequences = pad_sequences(sequences, maxlen=maxlen)
        self._size = len(self.sequences)
        
    def _gen_sequences(self, df, shuffle=True):
        """
        Generate sequences.
        If too long, divide it into multiple slice; 
        If too short, drop it.
        
        Return a generator: tuple (user, sequence)
        """
        for row in df.itertuples():
            uid, seq = row[1], row[2]

            # Skip short sequence
            if len(seq) < self.minlen: 
                continue

            for sub in sliding_windows(seq, window_size=self.maxlen, 
                                        step_size=self.maxlen):
                yield uid, sub

    def _get_negative_items(self, shape, prob=None):
        if prob is None:
            prob = np.ones(self._n_items)
        assert(len(prob) == self._n_items)
        
        prob /= prob.sum()

        negative_items = self.random_state.choice(np.arange(self._n_items),  p=prob,
                                                  size=shape)
        return negative_items
    
    def __iter__(self):
        """
        Iterate the whole dataset per mini-batch.
        """
        indices = np.arange(self._size)
        # Shuffle indices
        if self.shuffle:
            self.random_state.shuffle(indices)
        # Generate negative items
        negative_items = self._get_negative_items(shape=(self._size, self.maxlen), 
                                                  prob=self.sampling_prob)
        # per mini batch
        for i in range(0, self._size, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            
            yield self.sequences[batch_indices], negative_items[batch_indices]
