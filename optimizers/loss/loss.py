import random

import numpy as np
import math

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class LossObj:
    def __init__(self, data = [], batchSize = 1):
        self.data = data
        self.batchSize = batchSize
        self.amountOfDataVectors = 1
        self.currentBatchIndex = 0


        # Get the length of the indata. The other data should be of similar length.
        self.xDataLength = 1
        if len(self.data) > 0:
            # Pre-split the data (for calculating loss over the entire dataset)
            self.X = np.array(self.data[0])
            self.y = np.array(self.data[1])
            self.xDataLength = len(self.X[0])

            # Get the amount of data vectors (X,Y,...)
            self.amountOfDataVectors = len(self.data)

        self.numberOfBatches = math.ceil(self.xDataLength / self.batchSize)

        self.shuffledData = data # self.data.copy()

        # Allocate memory, the none lists will be replaced by numpy data vectors. Each batch starts with weights/posistion, output/labels, neural input/features.
        # [
        #    [[None], [None], [None]...],  # batch 0
        #    [[None], [None], [None]...]  # batch 1
        #    ...
        # ]
        self.randomBatchList = [[[None] for _1 in range (self.amountOfDataVectors)] for _ in range(self.numberOfBatches)] #Allocate memory
        self.randomIndexList = []
        self.fillRandomBatchList() if len(data) > 0 else None

    def minima(self):
        return None # calculate, if analytically possible, in each child lossObj class

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

    def getCurrentBatch(self):
        batch = self.randomBatchList[self.currentBatchIndex]
        X = batch[0] # first element vector is X
        y = batch[1] # second element vector is y
        return np.array(X), np.array(y)

    def fillRandomBatchList(self):
        if len(self.data) > 0:
            self.randomIndexList = np.random.permutation(self.xDataLength)

            # Shuffles the data in each data vector but keeps related values in appropriate places
            for i in range(self.amountOfDataVectors):
                self.shuffledData[i] = self.data[i][self.randomIndexList]

            # Fills the randomBatchList with batches from the shuffled data
            # 'i' will increase by batchsize each iteration and 'idx' will increase by 1 each time in the same iteration.
            idx = 0
            for i in range(0, self.xDataLength, self.batchSize):
                for j in range(self.amountOfDataVectors):
                    self.randomBatchList[idx][j] = self.shuffledData[j][i: i + self.batchSize]
                idx = idx + 1

            self.currentBatchIndex = 0
        return

#Test
if __name__ == "__main__":
    # Data is in format: np.array([x1, x2, x3...]), np.array([y1,y2,y3])
    testLoss = LossObj(data=[np.array([[4,5,6], [7,8,9], [10,11,12]]), np.array([1,2,3])], batchSize= 2)
    testLoss.fillRandomBatchList()
    print(testLoss.randomBatchList)
    