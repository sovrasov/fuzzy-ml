# -*- coding: utf-8 -*-

from miscFunctions import *
import numpy as np
import copy

def randomVectorConstrained(lBound, uBound):
    return np.array([random.uniform(lBound[i], uBound[i]) for i in range(len(lBound))])

def checkBounds(vector, bounds)
    if np.where(vector < bounds[0])[0].size != 0 and np.where(vector > bounds[1])[0].size != 0:
        return False
    return True

def PSO(objectiveFunction, firstPoint, bounds, numberOfParticles = 30):
    spaceDimension = len(firstPoint)
    swarm = [randomVectorConstrained(bounds[0], bounds[1]) for _ in range(numberOfParticles)]
    swarm.append(firstPoint)
    swarmBest = copy.deepcopy(swarm)
    bestValues = [objectiveFunction(x) for x in swarmBest]
    bestParam = copy.copy(firstPoint)
    bestGlobalValue = objectiveFunction(bestParam)
    velocities = np.array([[0.0] * spaceDimension] * (numberOfParticles + 1))
    maxIterations = 20
    iters = 0
    eps = 10e-4
    w = 0.3
    c1 = 0.2
    c2 = 0.5

    print('-'*50)
    print('Initial best falue:\t{}'.format(bestGlobalValue))

    while iters < maxIterations:
        for i in range(numberOfParticles):
            velocities[i] = w*velocities[i] + c1*random.uniform(0,1)* \
                    (swarmBest[i] - swarm[i]) + c2*random.uniform(0,1)*(bestParam-swarm[i])
            swarm[i] += velocities[i]
            #if not checkBounds(swarm[i], bounds)
            #    print('stepped out')
        for i in range(numberOfParticles):
            currentValue = objectiveFunction(swarm[i])
            if(currentValue < bestGlobalValue):
                bestGlobalValue = currentValue
                bestParam = copy.copy(swarm[i])
            if(currentValue < bestValues[i]):
                bestValues[i] = currentValue
                swarmBest[i] = copy.copy(swarm[i])

        print('New best falue: \t{}'.format(bestGlobalValue))
        iters += 1

    print('-'*50)
    return bestParam
