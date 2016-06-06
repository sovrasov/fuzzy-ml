# -*- coding: utf-8 -*-

import numpy as np
from miscFunctions import *

class TSK0():

    def __init__(self):
        self.centers = []
        self.vars = []
        self.b = []
        self.numberOfRules = 0
        self.inputDimension = 0
        self.numberOfClasses = 0

    def __evaluateConfidence(self, classID, x):
        expValues = [np.exp((-0.5*(x[i] - self.centers[classID][i])**2)/self.vars[classID]**2) for i in range(self.inputDimension)]
        return np.prod(expValues)

    def initFromClusters(self, clusterCenters, x, y, numberOfClasses):
        self.centers = clusterCenters
        self.numberOfRules = len(clusterCenters)
        self.inputDimension = len(x[0])
        self.numberOfClasses = numberOfClasses

        for i in range(self.numberOfRules):
            distances = []#[dist(self.centers[i], center) for center in self.centers]
            for j in range(self.numberOfRules):
                if j != i:
                    distances.append(dist(self.centers[i], self.centers[j]))
                else:
                    distances.append(float('inf'))
            h = np.argmin(distances)
            self.vars.append(distances[h] / 1.5)

        for i in range(self.numberOfRules):
            confidences = [self.__evaluateConfidence(i, vector) for vector in x]
            multiplicatedConfidences = [confidences[i]*y[i] for i in range(len(y))]
            self.b.append(np.sum(multiplicatedConfidences) / np.sum(confidences))

        print(self.b)

    def predict(self, x):
        #firstLayersOutput = 
        return 3

    def score(self, x, y):
        answers = [self.predict(vector) for vector in x]
        return np.where(np.array(answers) == np.array(y))[0].size / float(len(y))