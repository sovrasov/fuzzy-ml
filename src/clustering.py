#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

from miscFunctions import *
import numpy as np

def getKohonenClusters(vectors, numberOfClusters = 15, seed = 0):
    randomInstance = random.Random(seed)
    vecSize = len(vectors[0])
    eps = 10e-5
    maxIters = 20
    winsCounters = [1] * numberOfClusters
    alphaR = 0.01
    alphaW = 0.7

    clusterCenters = [np.array(random_floats(0., 1., vecSize, randomInstance)) \
        for _ in xrange(numberOfClusters)]
    oldClustersCenters = np.copy(clusterCenters)
    clustersDist = float('inf')
    iters = 0

    while iters < maxIters and clustersDist > eps:
        totalWins = np.sum(winsCounters)
        for vector in vectors:
            distances = [winsCounters[i] * dist(clusterCenters[i], vector) / totalWins \
                for i in xrange(numberOfClusters)]
            w = np.argmin(distances)
            distances[w] = float('inf')
            r = np.argmin(distances)
            clusterCenters[w] += alphaW * (vector - clusterCenters[w])
            clusterCenters[r] -= alphaR * (vector - clusterCenters[r])
            winsCounters[w] += 1

        clustersDist = 0.0
        for i in xrange(numberOfClusters):
            clustersDist += dist(clusterCenters[i], oldClustersCenters[i])
        clustersDist /= numberOfClusters
        oldClustersCenters = np.copy(clusterCenters)

        alphaW -= alphaW * iters / maxIters
        alphaR -= alphaR * iters / maxIters
        iters += 1

    filteredClusters = []
    for cluster in clusterCenters:
        if np.where(cluster < 0.0)[0].size == 0 and np.where(cluster > 1.0)[0].size == 0:
            filteredClusters.append(cluster)

    return filteredClusters
