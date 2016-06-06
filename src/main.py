# -*- coding: utf-8 -*-

import sys
from miscFunctions import *
from clustering import *
import numpy as np
from tsk0Model import TSK0

def main(args):
    dataSet = loadNormalizedData()
    print('Dataset loaded')

    clusterCenters = getKohonenClusters(dataSet[0])
    print('Clusters found:')
    for cluster in clusterCenters:
        print(cluster)

    print('Building model...')
    model = TSK0()
    model.initFromClusters(clusterCenters, dataSet[0], dataSet[1], 3)
    print('Score: {}'.format(model.score(dataSet[0], dataSet[1])))
    print('Optimizing model...')
    print('Score: {}'.format(model.score(dataSet[0], dataSet[1])))

if __name__ == '__main__':
    main(sys.argv)
