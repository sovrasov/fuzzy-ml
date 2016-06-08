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
    def __init__(self, job):
        threading.Thread.__init__(self)
        self.job = job
        self.result = 0
    def run(self):
        self.result = self.job()
        print('#'*60)
        print('Evaluated score {}'.format(self.result))

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
    random.seed(10)
    random.shuffle(data)
    slices = [data[i::k] for i in xrange(k)]

    scores = []
    threads = []

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        xTrain, yTrain = zip(*training)
        xTest, yTest = zip(*validation)
        thread = ThreadWorker(lambda: modelEvaluator(xTrain, yTrain, xTest, yTest))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        scores.append(thread.result)

    return np.sum(np.array(scores)) / k
