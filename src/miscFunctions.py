#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import random
import numpy as np

def printIf(ountput, needPrint = True):
    if needPrint:
        print(ountput)

def getTSK0Score(model, params, xTest, yTest):
    model.decode(params)
    error = 0.0
    for i in xrange(len(xTest)):
        error += (yTest[i] - model.predictRaw(xTest[i]))**2
    return error / len(xTest)

def splitDataset(x, y, testRatio = 0.2, seed = 0):
    randomInstance = random.Random(seed)
    data = zip(x, y)
    trainSize = int((1.0 - testRatio)*len(y))
    randomInstance.shuffle(data)
    xTrain, yTrain = zip(*data[:trainSize])
    xTest, yTest = zip(*data[trainSize:])
    return xTrain, yTrain, xTest, yTest

def random_floats(low, high, size, randomInstance):
    return [randomInstance.uniform(low, high) for _ in xrange(size)]

def dist(x, y, squared = True):
    if squared:
        return np.sum((x - y)**2)
    return np.sqrt(numpy.sum((x - y)**2))

def loadNormalizedData(path, vecSize, nameToDigit):
    x = []
    y = []
    lines = [line.rstrip('\n') for line in open(path)]

    xMin = [float("inf")] * vecSize
    xMax = [float("-inf")] * vecSize

    for line in lines:
        terms = line.split(',')
        if(len(terms) > 1):
            vector = [float(terms[i]) for i in xrange(vecSize)]
            x.append(vector)

            for i in xrange(vecSize):
                xMin[i] = min(x[-1][i], xMin[i])
                xMax[i] = max(x[-1][i], xMax[i])

            y.append(nameToDigit[terms[-1]])

    for vector in x:
        for i in xrange(vecSize):
            vector[i] = (vector[i] - xMin[i]) / (xMax[i] - xMin[i])

    return np.array(x), np.array(y)
