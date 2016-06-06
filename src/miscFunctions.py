# -*- coding: utf-8 -*-

import random
import numpy

def random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]

def dist(x, y, squared = True):
    if squared:
        return numpy.sum((x - y)**2)
    return numpy.sqrt(numpy.sum((x - y)**2))

def loadNormalizedData():

    vecSize = 4
    nameToDigit = {'Iris-virginica': 1, 'Iris-setosa': 2, 'Iris-versicolor': 3}
    x = []
    y = []
    lines = [line.rstrip('\n') for line in open('../data/iris.data')]

    xMin = [float("inf")] * vecSize
    xMax = [float("-inf")] * vecSize

    for line in lines:
        terms = line.split(',')
        if(len(terms) > 1):
            x.append([float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3])])

            for i in range(vecSize):
                xMin[i] = min(x[-1][i], xMin[i])
                xMax[i] = max(x[-1][i], xMax[i])

            y.append(nameToDigit[terms[4]])

    for vector in x:
        for i in range(vecSize):
            vector[i] = (vector[i] - xMin[i]) / (xMax[i] - xMin[i])

    return x, y
