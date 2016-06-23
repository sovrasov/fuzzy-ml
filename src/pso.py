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
import copy

def randomVectorConstrained(lBound, uBound, rndInstance):
    return np.array([rndInstance.uniform(lBound[i], uBound[i]) \
        for i in xrange(len(lBound))])

def checkBounds(vector, bounds):
    if np.all(vector > bounds[0]) and np.all(vector < bounds[1]):
        return True
    return False

def PSO(objectiveFunction, firstPoint, bounds, numberOfParticles = 100,
        verbose = True, seed = 0):
    rndInstance = random.Random(seed)
    spaceDimension = len(firstPoint)
    swarm = [randomVectorConstrained(bounds[0], bounds[1], rndInstance) \
        for _ in xrange(numberOfParticles - 1)]
    if not checkBounds(firstPoint, bounds):
        firstPoint = randomVectorConstrained(bounds[0], bounds[1], rndInstance)
    swarm.append(np.array(firstPoint))
    swarmBest = copy.deepcopy(swarm)
    bestValues = [objectiveFunction(x) for x in swarmBest]
    bestParam = copy.copy(firstPoint)
    bestGlobalValue = objectiveFunction(bestParam)
    lastBestGlobalValue = bestGlobalValue*2
    velocities = np.array([[0.0] * spaceDimension] * (numberOfParticles))
    maxIterations = 30
    iters = 0
    eps = 10e-3
    delta = 10e-5
    w = 0.8
    c1 = 0.3
    c2 = 0.9

    printIf('-'*50, verbose)
    printIf('Initial best value:\t{}'.format(bestGlobalValue), verbose)

    while iters < maxIterations and bestGlobalValue > eps and \
        lastBestGlobalValue - bestGlobalValue > delta:
        for i in xrange(numberOfParticles):
            velocities[i] = w*velocities[i] + c1*rndInstance.uniform(0.0,1.0)* \
                    (swarmBest[i] - swarm[i]) + \
                    c2*rndInstance.uniform(0.0,1.0)*(bestParam - swarm[i])
            if not checkBounds(swarm[i] + velocities[i], bounds):
                while True:
                    velocities[i] /= 2
                    if checkBounds(swarm[i] + velocities[i], bounds):
                        break
            swarm[i] += velocities[i]

        for i in xrange(numberOfParticles):
            currentValue = objectiveFunction(swarm[i])
            if(currentValue < bestGlobalValue):
                printIf('New best value: \t{}'.format(bestGlobalValue), verbose)
                lastBestGlobalValue = bestGlobalValue
                bestGlobalValue = currentValue
                bestParam = copy.copy(swarm[i])
            if(currentValue < bestValues[i]):
                bestValues[i] = currentValue
                swarmBest[i] = copy.copy(swarm[i])

        iters += 1

    printIf('-'*50, verbose)
    printIf('PSO iterations: \t{}'.format(iters), verbose)
    printIf('-'*50, verbose)
    return bestParam
