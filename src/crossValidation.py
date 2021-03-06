#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import multiprocessing as mp
import numpy as np

from miscFunctions import *
from clustering import getKohonenClusters
from tsk0Model import TSK0
from pso import PSO

def buildAndTestModel(args, xTrain, yTrain, xTest, yTest, conn):
    clusterCenters = getKohonenClusters(xTrain, args.nClusters, args.seed)
    model = TSK0()
    model.initFromClusters(clusterCenters, xTrain, yTrain)
    initialParams = model.code()
    newParams = PSO(lambda x: getTSK0Score(model, x, xTrain, yTrain),
    model.code(), model.getParametersBounds(), args.nParticles, False, args.seed)
    model.decode(newParams)
    conn.send(model.score(xTest, yTest))

def getTSK0KFoldCVScore(modelEvaluator, x, y, k = 5, seed = 0):
    rndInstance = random.Random(seed)
    data = zip(x, y)
    rndInstance.shuffle(data)
    scores = []
    threads = []

    for i in xrange(k):
        training = [x for j, x in enumerate(data) if j % k != i]
        validation = [x for j, x in enumerate(data) if j % k == i]
        xTrain, yTrain = zip(*training)
        xTest, yTest = zip(*validation)
        parentPipe, childPipe = mp.Pipe()
        thread = mp.Process(target=modelEvaluator, args=\
            (xTrain, yTrain, xTest, yTest, childPipe))
        thread.start()
        threads.append([thread, parentPipe])

    for i in xrange(len(threads)):
        currentScore = threads[i][1].recv()
        threads[i][0].join()
        scores.append(currentScore)
        printIf('Score on split {}: \t{}'.format(i + 1, scores[-1]))

    return np.sum(scores) / k, np.std(scores)
