import torch
import torch.utils.data

import numpy as np
from scipy.sparse import csr_matrix

import csv
from time import time


class DataSet():
    """ Data set
    x:    input
    y:    target
    mask: sparse matrix mask
    """
    def __init__(self, x, y=None, mask=None):
        self.x = x
        self.y = y
        self.mask = mask
        
    def batch_by_index(self, indices):
        """
        Get batch by indices.
        """
        x = self.x[indices,:]
        y = self.y[indices,:] if self.y is not None else None
        mask = self.mask[indices,:] if self.mask is not None else None
        return DataSet(x, y=y, mask=mask)
        
    def extract(self, i, size=1):
        """
        i:    start position
        size: selected chunk size
        """
        if i>= self.__len__() or i < 0:
            raise IndexError("Out of range")
        x = self.x[i:i+size,:]
        y = self.y[i:i+size,:] if self.y is not None else None
        mask = self.mask[i:i+size,:] if self.mask is not None else None
        return DataSet(x, y=y, mask=mask)
        
    def __len__(self):
        return self.x.shape[0]
    
    @property
    def width(self):
        return self.x.shape[1]
    
    @property
    def size(self):
        return self.mask.data.size
    
    
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

        return rows, cols, ratings
    
    def _array_to_matrix(self, rows, cols, ratings, isTest=False):
        """ Convert to sparse(csr_matrix)
        """
        if isTest:
            # Re-assignment rows
            height = np.unique(rows).size
            newRows = np.zeros(len(rows))
            cnt = 0
            for i in range(1, len(rows)):
                if rows[i] != rows[i-1]:
                    cnt += 1
                newRows[i] = cnt
            rows = newRows
        else:
            height = len(self.user_hash)
            
        self.width = width = len(self.item_hash)
        
        # Create sparse data matrix.
        data = np.array(ratings, dtype='float32')
        mat = csr_matrix((data, (rows, cols)), shape=(height, width))
        # Create mask
        data = np.ones(len(ratings))
        mask = csr_matrix((data, (rows, cols)), shape=(height, width))
        
        return mat, mask
    
    def load(self, trainfile, testfile=None):
        """ Load csvfile and return dataset.
        trainfile: csvfile path
        testfile : csvfile path
        """
        start = time()
        print("Loading train set ... ", end='', flush=True)
        traindata = self._load(trainfile)
        print("{0:.2f}s".format(time()-start))
        print("Loading test set ... ", end='', flush=True)
        start = time()
        testdata  = self._load(testfile)
        print("{0:.2f}s".format(time()-start))
        
        # Create trainset
        rows, cols, ratings = traindata
        trainmat, trainmask = self._array_to_matrix(rows, cols, ratings)
        
        mean = trainmat.sum(1) / trainmask.sum(1)  # Mean of rows (users).
        trainmat -= trainmask.multiply(mean)       # Remove mean.
        
        trainset = DataSet(x=trainmat, 
                           y=trainmat, 
                           mask=trainmask)
        print("Trainset: \t {} users \t {} items".format(len(trainset), trainset.width))
        
        # Create testset
        rows, cols, ratings = testdata
        testmat, testmask = self._array_to_matrix(rows, cols, ratings, isTest=True)
        
        test_users = np.unique(rows)                     # Extract test users
        testmat -= testmask.multiply(mean[test_users,:]) # Remove mean of test from train.
        
        testset = DataSet(x=trainmat[test_users,:], 
                          y=testmat,
                          mask=testmask)
        print("Testset: \t {} users \t {} items".format(len(testset), testset.width))
        
        return trainset, testset

class MiniBatcher():
    def __init__(self, dataset, batchSize=3):
        """
        dataset: DataSet
        """
        self.dataset = dataset
        self.batchSize = batchSize
        
        self.shuffleIdx = np.random.permutation(len(dataset))
        self.idxIter = iter(self._nextBatchIdx())
    
    def _nextBatchIdx(self):
        """Yield idx in batch size."""
        for i in range(0, len(self.shuffleIdx), self.batchSize):
            yield self.shuffleIdx[i:i+self.batchSize]
            
    def __next__(self):
        try:
            nextIdx = next(self.idxIter)
        except:
            raise StopIteration
        
        return self.dataset.batch_by_index(nextIdx)
                
    def __iter__(self):
        return self
        
    def __len__(self):
        return len(self.dataset)