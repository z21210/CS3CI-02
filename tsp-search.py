import numpy as np
from enum import Enum
from random import shuffle
import csv, time, sys
import math as maths
import matplotlib.pyplot as plt
import networkx as nx

Cities = {}
with open('ulysses16.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        Cities[row['id']] = [float(row['x']), float(row['y'])]
CITIES = list(Cities.keys())

City = Enum('City', [[CITIES[i], i] for i in range(len(CITIES))])

def getEuclideanDistance(fromCity, toCity):
    fromName = City(fromCity).name
    toName = City(toCity).name
    fromX, fromY = Cities[fromName]
    toX, toY = Cities[toName]
    return maths.sqrt((toX - fromX)**2 + (toY - fromY)**2)


CONNECTIONS = np.zeros((len(CITIES), len(CITIES)))
for y in range(0, len(CITIES)):
    for x in range(0, len(CITIES)):
        CONNECTIONS[y][x] = getEuclideanDistance(y, x)

plt.ion()
graph = nx.DiGraph()
graph.add_nodes_from(CITIES)
def drawRoute(route, cost, status='Running...'):
    if (len(sys.argv) < 4):
        return
    graph.clear_edges()
    fromIndex = 0
    for toIndex in route:
        fromCity = City(fromIndex).name
        toCity = City(toIndex).name
        graph.add_edge(fromCity, toCity)
        fromIndex = toIndex
    graph.add_edge(City(fromIndex).name, City(0).name, weight=getDistance(fromIndex, 0))
    plt.clf()
    plt.suptitle(f'Route: {formatRoute(route)}\nCost: {cost}\n{status}')
    nx.draw_networkx(graph, pos=Cities, with_labels=True)
    plt.show()
    plt.pause(0.001)


def getDistance(fromCity, toCity):
    return CONNECTIONS[fromCity, toCity]

def generateRoute():
    route = list(range(1,len(CITIES)))
    shuffle(route)
    return route

def evaluateRoute(route):
    totalDistance = 0
    totalDistance += getDistance(0, route[0])
    for i in range(0, len(route) - 1):
        fromCity = route[i]
        toCity = route[i+1]
        distance = getDistance(fromCity, toCity)
        totalDistance += distance
    totalDistance += getDistance(route[len(route)-1], 0)
    return totalDistance

def getBestRoute(routes):
    bestRoute, bestCost = None, None
    for route in routes:
        cost = evaluateRoute(route)
        if (bestCost == None) or (cost < bestCost):
            bestRoute, bestCost = route, cost
    return bestRoute


def formatRoute(route):
    formatted = f'{City(0).name} -> '
    for i in range(0, len(route)):
        formatted +=  f'{City(route[i]).name} -> '
    formatted += f'{City(0).name}'
    return formatted

def randomSearch(timeLimit):
    start, now = time.time(), time.time()
    end = start + timeLimit
    bestRoute, bestCost = None, None
    while time.time() < end:
        print(f'Time left: {int(end - time.time())}s     ', end='\r')
        route = generateRoute()
        cost = evaluateRoute(route)
        if (bestCost == None) or (cost < bestCost):
            bestRoute, bestCost = route, cost
            drawRoute(route, cost)
    return bestRoute, bestCost

def twoOpt(route, node1Index, node2Index):
    if node2Index < node1Index:
        node1Index, node2Index = node2Index, node1Index
    start = route[0:node1Index]
    middle = route[node1Index:node2Index+1]
    middle.reverse()
    end = route[node2Index+1:len(route)]
    return start + middle + end

def getTwoOpts(route):
    twoOpts = []
    for i in range(0, len(route)-1):
        for j in range(i+1, len(route)):
            twoOpts.append(twoOpt(route, i, j))
    return twoOpts

def bestNeighbourSearch(timeLimit):
    start, now = time.time(), time.time()
    end = start + timeLimit
    route = generateRoute()
    while time.time() < end:
        print(f'Time left: {int(end - time.time())}s     ', end='\r')
        bestNeighbour = getBestRoute([route] + getTwoOpts(route))
        drawRoute(bestNeighbour, evaluateRoute(bestNeighbour))
        if bestNeighbour == route:
            print('\nLocal optimum found. Terminating early.')
            break # found local optimum - terminate
        route = bestNeighbour

    return route, evaluateRoute(route)

def bestDeepNeighbourSearch(timeLimit):
    start, now = time.time(), time.time()
    end = start + timeLimit
    route = generateRoute()
    while time.time() < end:
        print(f'Time left: {int(end - time.time())}s     ', end='\r')
        neighbours = getTwoOpts(route)
        bestDeepNeighbours = []
        for neighbour in neighbours:
            bestDeepNeighbours.append(getBestRoute([neighbour] + getTwoOpts(neighbour)))
        bestDeepNeighbour = getBestRoute(bestDeepNeighbours)
        drawRoute(bestDeepNeighbour, evaluateRoute(bestDeepNeighbour))
        if bestDeepNeighbour == route:
            print('\nLocal optimum found. Terminating early.')
            break # found local optimum - terminate
        route = bestDeepNeighbour
    return route, evaluateRoute(route)

def bestNeighbourSearchWithPerturbations(timeLimit):
    start, now = time.time(), time.time()
    end = start + timeLimit
    route = generateRoute()
    while time.time() < end:
        print(f'Time left: {int(end - time.time())}s     ', end='\r')
        bestNeighbour = getBestRoute([route] + getTwoOpts(route))
        if bestNeighbour == route:
            #found local optima - introduce random competition
            randomRoute = generateRoute()
            bestNeighbour = getBestRoute([route, randomRoute] + getTwoOpts(randomRoute))
        route = bestNeighbour
        drawRoute(bestNeighbour, evaluateRoute(bestNeighbour))
    return route, evaluateRoute(route)



def main():
    if len(sys.argv) < 3:
        sys.exit('Missing argument. Required positional arguments: mode, time limit.')
    try:
        timeLimit = float(sys.argv[2])
    except ValueError:
        sys.exit('Time limit must be int or float')
    mode = sys.argv[1]
    if mode == 'random':
        route, cost = randomSearch(timeLimit)
    elif mode == 'best':
        route, cost = bestNeighbourSearch(timeLimit)
    elif mode == 'deep':
        route, cost = bestDeepNeighbourSearch(timeLimit)
    elif mode == 'perturb':
        route, cost = bestNeighbourSearchWithPerturbations(timeLimit)
    else:
        sys.exit('Modes are: random, best, deep, perturb.')
    print(f'Route: {formatRoute(route)}\nCost: {cost}')
    plt.ioff()
    drawRoute(route, cost, 'Done.')

main()