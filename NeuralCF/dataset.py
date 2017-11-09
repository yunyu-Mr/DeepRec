import csv
# import pandas as pd
import numpy as np


class DataManager():
    """Encapsulates low-level data loading.
    """
    def __init__(self):
        """Initialize loader
        width: the possible number of bits, which is the dimensionality of the vectors
        """
        self.width = 0
        self.user_hash = {}
        self.item_hash = {}  # maps from word to vector index

    def index_for_item(self, iid):
        """returns a list of k indices into the output vector
        corresponding to the bits for this word
        """
        if not iid in self.item_hash:
            idx = len(self.item_hash)
            self.item_hash[iid] = idx
        return self.item_hash[iid]

    def index_for_user(self, uid):
        if not uid in self.user_hash:
            idx = len(self.user_hash)
            self.user_hash[uid] = idx
        return self.user_hash[uid]

    def _load(self, csvfile):
        """ Load data file and create validation set. """
        rows = []           # row indices
        cols = []           # col indices
        ratings = []  # Ratings of trainset

        # Load train file.
        with open(csvfile, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row['user']
                u = self.index_for_user(uid)
                iid = row['item']
                i = self.index_for_item(iid)

                rows.append(u)
                cols.append(i)
                rating = (float(row['rating']) - 3) / 2
                ratings.append(rating)

        return np.array(rows), np.array(cols), np.array(ratings, dtype='float32')

    def load_dataset(self, trainfile, testfile=None):
        """
        Load csvfile and wrap it into dataset (spotlight Interactions)

        Input: csvfile path

        Return: trainset and testset (spotlight.interactions.Interactions)
        """
        from spotlight.interactions import Interactions

        traindata = self._load(trainfile)
        if testfile is not None:
            testdata  = self._load(testfile)

        num_users = len(self.user_hash)
        num_items = len(self.item_hash)

        trainset = Interactions(*traindata, 
                        num_items=num_items, 
                        num_users=num_users)
        if testfile is not None:
            testset  = Interactions(*testdata,
                            num_items=num_items,
                            num_users=num_users)
        else: testset = None

        return trainset, testset