# -*- coding: utf-8 -*-

from miscFunctions import *
import numpy as np

def getKohonenClusters(vectors, numberOfClusters = 10):
    vecSize = len(vectors[0])
    eps = 10e-4
    maxIters = 10
    winsCounters = [1] * numberOfClusters
    alphaR = 0.01
    alphaW = 0.6

    clusterCenters = [np.array(random_floats(0., 1., vecSize)) for _ in range(numberOfClusters)]
    oldClustersCenters = np.copy(clusterCenters)
    clustersDist = float('inf')
    iters = 0

    while iters < maxIters and clustersDist > eps:
        totalWins = np.sum(winsCounters)
        for vector in vectors:
            distances = [winsCounters[i] * dist(clusterCenters[i], vector) / totalWins for i in range(numberOfClusters)]
            #print(distances)
            w = np.argmin(distances)
            distances[w] = float('inf')
            r = np.argmin(distances)
            #if r == w:
            #    print('error')
            clusterCenters[w] += alphaW * (vector - clusterCenters[w])
            clusterCenters[r] -= alphaR * (vector - clusterCenters[r])
            winsCounters[w] += 1

        clustersDist = 0.0
        for i in range(numberOfClusters):
            clustersDist += dist(clusterCenters[i], oldClustersCenters[i])
#            print(clusterCenters[i])
#            print(oldClustersCenters[i])
        clustersDist /= numberOfClusters
        oldClustersCenters = np.copy(clusterCenters)

        alphaW -= alphaW * iters / maxIters
        alphaR -= alphaR * iters / maxIters
        iters += 1
#        print('cluster dist {}'.format(clustersDist))
    filteredClusters = []
    for cluster in clusterCenters:
        if np.where(cluster < 0.0)[0].size == 0 and np.where(cluster > 1.0)[0].size == 0:
            filteredClusters.append(cluster)

    #print('Iterations: {}'.format(iters))
    return filteredClusters
