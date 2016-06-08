#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import threading
import numpy as np

from miscFunctions import *
from clustering import getKohonenClusters
from tsk0Model import TSK0
from pso import PSO

class ThreadWorker (threading.Thread):
    def __init__(self, job, ID):
        threading.Thread.__init__(self)
        self.job = job
        self.result = 0.0
        self.ID = ID
    def run(self):
        self.result = self.job()

def buildAndTestModel(args, xTrain, yTrain, xTest, yTest):
    random.seed(args.seed)
    clusterCenters = getKohonenClusters(xTrain, args.nClusters)
    model = TSK0()
    model.initFromClusters(clusterCenters, xTrain, yTrain)
    initialParams = model.code()
    newParams = PSO(lambda x: getTSK0Score(model, x, xTrain, yTrain),
        model.code(), model.getParametersBounds(), args.nParticles, False)
    model.decode(newParams)

    return model.score(xTest, yTest)

def getTSK0KFoldCVScore(modelEvaluator, x, y, k=5):
    data = zip(x, y)
    random.shuffle(data)
    slices = [data[i::k] for i in xrange(k)]

    score = 0.0
    threads = []

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        xTrain, yTrain = zip(*training)
        xTest, yTest = zip(*validation)
        thread = ThreadWorker(lambda: modelEvaluator(xTrain, yTrain, xTest, yTest), i + 1)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        score += thread.result
        printIf('Quality on split {}: \t{}'.format(thread.ID, thread.result))

    return score / k
