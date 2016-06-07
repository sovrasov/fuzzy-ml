#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
from miscFunctions import *

class TSK0():

    def __init__(self):
        self.centers = []
        self.vars = []
        self.b = []
        self.numberOfRules = 0
        self.inputDimension = 0

    def __evaluateConfidence(self, ruleID, x):
        expValues = [np.exp(-0.5*((x[i] - self.centers[ruleID][i]) / self.vars[ruleID][i])**2)
                     for i in xrange(self.inputDimension)]
        return np.prod(expValues)

    def getParametersBounds(self):
        lBound = []
        uBound = []

        for i in xrange(self.numberOfRules):
            lBound.extend([0.0]*self.inputDimension)
            uBound.extend([1.0]*self.inputDimension)

        lBound.extend([0.0] * ((1 + self.inputDimension) * self.numberOfRules))
        uBound.extend([5.0] * ((1 + self.inputDimension) * self.numberOfRules))

        return lBound, uBound

    def code(self):
        parameters = []
        for center in self.centers:
            parameters.extend(center)
        for var in self.vars:
            parameters.extend(var)
        parameters.extend(self.b)
        return parameters

    def decode(self, parameters):
        for i in xrange(self.numberOfRules):
            self.centers[i] = parameters[i*self.inputDimension : (i+1)* \
                    self.inputDimension]
        offset = self.numberOfRules*self.inputDimension
        for i in xrange(self.numberOfRules):
            self.vars[i] = parameters[offset + i*self.inputDimension : (i+1)* \
                    self.inputDimension + offset]
        self.b = parameters[self.numberOfRules*self.inputDimension*2 :]

    def initFromClusters(self, clusterCenters, x, y):
        self.centers = clusterCenters
        self.numberOfRules = len(clusterCenters)
        self.inputDimension = len(x[0])

        for i in xrange(self.numberOfRules):
            distances = []
            for j in xrange(self.numberOfRules):
                if j != i:
                    distances.append(dist(self.centers[i], self.centers[j]))
                else:
                    distances.append(float('inf'))
            h = np.argmin(distances)
            self.vars.append([distances[h] / 1.5]*self.inputDimension)

        for i in xrange(self.numberOfRules):
            confidences = [self.__evaluateConfidence(i, vector) for vector in x]
            multiplicatedConfidences = np.multiply(confidences, y)
            self.b.append(np.sum(multiplicatedConfidences) / np.sum(confidences))

    def predictRaw(self, x):
        firstLayersOutput = [self.__evaluateConfidence(i, x) for i in xrange(self.numberOfRules)]
        sum2 = np.sum(firstLayersOutput)
        sum1 = np.sum(np.multiply(firstLayersOutput, self.b))
        return sum1 / sum2

    def predict(self, x):
        return np.ceil(self.predictRaw(x) - 0.5)

    def score(self, x, y):
        answers = [self.predict(vector) for vector in x]
        return np.where(np.array(answers) == np.array(y))[0].size / float(len(y))
