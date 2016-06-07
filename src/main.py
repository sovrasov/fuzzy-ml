#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import argparse
import numpy as np

from miscFunctions import *
from clustering import getKohonenClusters
from tsk0Model import TSK0
from pso import PSO

def main():
    parser = argparse.ArgumentParser(description='Building and optimization of \
            TSK0 model for sovling irises classification problem')
    parser.add_argument('-nc', '--nClusters', help='Number of clusters in Kohonen network', \
            type=int, default=10)
    parser.add_argument('-np', '--nParticles', help='Number of particles in PSO', \
            type=int, default=15)
    parser.add_argument('-s', '--seed', help='Seed for RNG', \
            type=int, default=100)
    parser.add_argument('-ts', '--testSize', type=float, default=0.3, \
            help = 'Relative size of test dataset')
    args = parser.parse_args()

    nameToDigit = {'Iris-virginica': 1, 'Iris-setosa': 2, 'Iris-versicolor': 3}
    dataSet = loadNormalizedData(4, nameToDigit)
    print('Dataset loaded')
    xTrain, yTrain, xTest, yTest = splitDataset(dataSet[0], dataSet[1], args.testSize)
    clusterCenters = getKohonenClusters(dataSet[0], args.nClusters)
    print('Clusters found: {}'.format(len(clusterCenters)))

    print('Building model...')
    model = TSK0()
    model.initFromClusters(clusterCenters, xTrain, yTrain)
    print('Testing model...')
    print('Train score: {}'.format(model.score(xTrain, yTrain)))
    print('Test score: {}'.format(model.score(xTest, yTest)))
    print('Optimizing model...')
    initialParams = model.code()
    newParams = PSO(lambda x: getTSK0Score(model, x, xTrain, yTrain),
        model.code(), model.getParametersBounds(), args.nParticles)
    model.decode(newParams)
    print('Testing model...')
    print('Train score: {}'.format(model.score(xTrain, yTrain)))
    print('Test score: {}'.format(model.score(xTest, yTest)))

if __name__ == '__main__':
    main()
