import random

import numpy as np
import math

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class LossObj:
    def __init__(self, data = None, batchAmount = 1):
        self.data = np.array(data) if data is not None else np.array([])
        self.batchAmount = batchAmount
        self.amountOfDataVectors = 1

        #Get the length of the indata. The other data should be of similar length.
        self.xDataLength = 1
        if len(self.data) > 0:
            self.xDataLength = len(self.data[0])

            #Get the amount of data vectors (X,Y,...)
            self.amountOfDataVectors = len(self.data)

        self.numberOfBatches = math.ceil(self.xDataLength / self.batchAmount)

        self.shuffledData = data


        #Allocate memory, the none lists will be replaced by numpy data vectors.
        #[
        #    [[None], [None], [None]...],  # batch 0
        #    [[None], [None], [None]...]  # batch 1
        #    ...
        #]
        self.randomBatchList = [[[None] for _1 in range (self.amountOfDataVectors)] for _ in range(self.numberOfBatches)] #Allocate memory
        self.randomIndexList = []
        self.currentBatch = 0


    def minima(self):
        return None # calculate if possible

    def evaluate_loss(self, position):
        """
        Position is the a numpy array where the loss should be evaluated
        """
        return np.array([])
    def evaluate_gradient(self, position):
        """
        Position is the a numpy array where the gradient should be evaluated
        """
        return np.array([])

    def fillRandomBatchList(self):

        self.randomIndexList = np.random.permutation(self.xDataLength)

        #Shuffles the data in each data vector but keeps related values in appropriate places
        for i in range(self.amountOfDataVectors):
            self.shuffledData[i] = self.data[i][self.randomIndexList]


        #Fills the randomBatchList with batches from the shuffled data
        for i in range(0, self.xDataLength, self.batchAmount):
            for j in range(self.amountOfDataVectors):
                self.randomBatchList[i][j] = self.shuffledData[j][i: i + self.batchAmount]

        return
