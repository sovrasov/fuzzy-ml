#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
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

    def __evaluateTermsConfidences(self, ruleID, x):
        expValues = [np.exp(-0.5*((x[i] - self.centers[ruleID][i]) / self.vars[ruleID][i])**2)
                     for i in xrange(self.inputDimension)]
        return expValues

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

    def error(self, x, y):
        errors = [(self.predictRaw(x[i]) - y[i])**2 for i in xrange(len(x))]
        return np.sum(errors)

    def predict(self, x):
        return np.ceil(self.predictRaw(x) - 0.5)

    def score(self, x, y):
        answers = [self.predict(vector) for vector in x]
        return np.where(np.array(answers) == np.array(y))[0].size / float(len(y))

    def fitWithGradient(self, x, y):
        delta = 10e-15
        maxIters = 200
        freq = 8
        eta = 0.001
        eps = 10e-8
        bestValue = self.error(x, y)
        for iter in xrange(maxIters):
            for t in xrange(len(x)):
                secondLayerOutput = [self.__evaluateTermsConfidences(i, x[t]) for i in xrange(self.numberOfRules)]
                thirdLayerOutput = [np.prod(ruleConf) for ruleConf  in secondLayerOutput]
                sum2 = np.sum(thirdLayerOutput)
                sum1 = np.sum(np.multiply(thirdLayerOutput, self.b))
                prediction = sum1 / sum2
                delta41 = 2. * (prediction - y[t]) / sum2
                delta42 = - 2. * (prediction - y[t]) * sum1 / (sum2**2 + delta)
                delta3 = [self.b[i] * delta41 + delta42 for i in xrange(self.numberOfRules)]
                delta2 = [[delta3[i]*thirdLayerOutput[i] / (secondLayerOutput[i][j] + delta) for j in xrange(self.inputDimension)] for i in xrange(self.numberOfRules)]
                deltaB = [delta41*self.b[i]*thirdLayerOutput[i] for i in xrange(self.numberOfRules)]
                deltaC = [[delta2[i][j]*secondLayerOutput[i][j]* \
                    (x[t][j] - self.centers[i][j]) / (self.vars[i][j]**2 + delta) for j in xrange(self.inputDimension)] for i in xrange(self.numberOfRules)]
                deltaA = [[delta2[i][j]*secondLayerOutput[i][j]* \
                    (x[t][j] - self.centers[i][j]) ** 2 / (self.vars[i][j]**3 + delta) for j in xrange(self.inputDimension)] for i in xrange(self.numberOfRules)]
                self.b -= eta*np.array(deltaB)
                for i in xrange(self.numberOfRules):
                    self.centers[i] -= eta*np.array(deltaC[i])
                    self.vars[i] -= eta*np.array(deltaA[i])
                eta = (1.0 - float(iter) / maxIters)*0.001
            if iter % freq == freq - 1:
                currentValue = self.error(x, y)
                if (np.abs(currentValue - bestValue) < eps ) or currentValue > bestValue:
                    break
                elif currentValue < bestValue:
                    bestValue = currentValue
