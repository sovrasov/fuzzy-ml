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

def randomVectorConstrained(lBound, uBound):
    return np.array([random.uniform(lBound[i], uBound[i]) for i in xrange(len(lBound))])

def checkBounds(vector, bounds):
    if np.where(vector < bounds[0])[0].size != 0 or np.where(vector > bounds[1])[0].size != 0:
        return False
    return True

def PSO(objectiveFunction, firstPoint, bounds, numberOfParticles = 100, verbose = True):
    spaceDimension = len(firstPoint)
    swarm = [randomVectorConstrained(bounds[0], bounds[1]) \
        for _ in xrange(numberOfParticles - 1)]
    swarm.append(np.array(firstPoint))
    swarmBest = copy.deepcopy(swarm)
    bestValues = [objectiveFunction(x) for x in swarmBest]
    bestParam = copy.copy(firstPoint)
    bestGlobalValue = objectiveFunction(bestParam)
    velocities = np.array([[0.0] * spaceDimension] * (numberOfParticles))
    maxIterations = 20
    iters = 0
    eps = 10e-3
    w = 0.4
    c1 = 0.3
    c2 = 0.7

    printIf('-'*50, verbose)
    printIf('Initial best value:\t{}'.format(bestGlobalValue), verbose)

    while iters < maxIterations and bestGlobalValue > eps:
        for i in xrange(numberOfParticles):
            velocities[i] = w*velocities[i] + c1*random.uniform(0.0,1.0)* \
                    (swarmBest[i] - swarm[i]) + c2*random.uniform(0.0,1.0)*(bestParam - swarm[i])
            if not checkBounds(swarm[i] + velocities[i], bounds):
                while True:
                    velocities[i] /= 2
                    if checkBounds(swarm[i] + velocities[i], bounds):
                        break
            swarm[i] += velocities[i]

        for i in xrange(numberOfParticles):
            currentValue = objectiveFunction(swarm[i])
            if(currentValue < bestGlobalValue):
                bestGlobalValue = currentValue
                bestParam = copy.copy(swarm[i])
            if(currentValue < bestValues[i]):
                bestValues[i] = currentValue
                swarmBest[i] = copy.copy(swarm[i])

        printIf('New best value: \t{}'.format(bestGlobalValue), verbose)
        iters += 1

    printIf('-'*50, verbose)
    return bestParam
