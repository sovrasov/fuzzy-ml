# -*- coding: utf-8 -*-

import sys
from miscFunctions import *
from clustering import *
import numpy as np
from tsk0Model import TSK0

def main(args):
    random.seed(100)
    dataSet = loadNormalizedData()
    print('Dataset loaded')
    xTrain, yTrain, xTest, yTest = splitDataset(dataSet[0], dataSet[1])

    clusterCenters = getKohonenClusters(dataSet[0])
    print('Clusters found:')
    for cluster in clusterCenters:
        print(cluster)

    print('Building model...')
    model = TSK0()
    model.initFromClusters(clusterCenters, xTrain, yTrain, 3)
    print('Testing model...')
    print('Train score: {}'.format(model.score(xTrain, yTrain)))
    print('Test score: {}'.format(model.score(xTest, yTest)))
    print('Optimizing model...')

    print('Testing model...')
    print('Score: {}'.format(model.score(dataSet[0], dataSet[1])))

if __name__ == '__main__':
    main(sys.argv)
