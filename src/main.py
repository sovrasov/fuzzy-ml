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

from crossValidation import *
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
            type=int, default=20)
    parser.add_argument('-s', '--seed', help='Seed for RNG', \
            type=int, default=1)
    parser.add_argument('-ts', '--testSize', type=float, default=0.2, \
            help = 'Relative size of test data set')
    parser.add_argument('-dp', '--dataPath', type=str, default='../data/iris.data', \
            help = 'Path to file with the iris dataset')
    parser.add_argument('-vm', '--validationMethod', type=str, default='oneshot', \
            help = 'Validation method', choices=[str('oneshot'), str('crossv')])
    parser.add_argument('-k', '--foldsNumber', type=int, default=4, \
            help = 'Number of folds in cross-validation')
    args = parser.parse_args()

    nameToDigit = {'Iris-virginica': 1, 'Iris-setosa': 2, 'Iris-versicolor': 3}
    dataSet = loadNormalizedData(args.dataPath, 4, nameToDigit)
    printIf('Dataset loaded')

    if (args.validationMethod == str('oneshot')):
        random.seed(args.seed)
        xTrain, yTrain, xTest, yTest = splitDataset(dataSet[0], dataSet[1], \
            args.testSize, args.seed)
        clusterCenters = getKohonenClusters(xTrain, args.nClusters, args.seed)
        printIf('Clusters found: {}'.format(len(clusterCenters)))
        printIf('Building model...')
        model = TSK0()
        model.initFromClusters(clusterCenters, xTrain, yTrain)
        printIf('Testing model...')
        printIf('Train score: {}'.format(model.score(xTrain, yTrain)))
        printIf('Test score: {}'.format(model.score(xTest, yTest)))
        printIf('Optimizing model...')
        initialParams = model.code()
        newParams = PSO(lambda x: getTSK0Score(model, x, xTrain, yTrain),
            model.code(), model.getParametersBounds(), args.nParticles, args.seed)
        model.decode(newParams)
        printIf('Testing model...')
        printIf('Train score: {}'.format(model.score(xTrain, yTrain)))
        printIf('Test score: {}'.format(model.score(xTest, yTest)))
    elif(args.validationMethod == str('crossv')):
        printIf('Start cross-validation...')
        score, stdDev = getTSK0KFoldCVScore( \
            lambda xTrain, yTrain, xTest, yTest, conn: buildAndTestModel( \
            args, xTrain, yTrain, xTest, yTest, conn), dataSet[0], dataSet[1],\
            args.foldsNumber)
        printIf('Cross-validation score: {}, standard deviation: {}'.format(score, stdDev))

if __name__ == '__main__':
    main()
