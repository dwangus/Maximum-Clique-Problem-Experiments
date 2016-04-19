import math
import time
from operator import itemgetter
import threading

#STILL TOO SLOW -- in fact, even slower than before!

def distance(point_list1, point_list2):
        temp = 0.0
        for i in range(len(point_list1)):
            temp += (float(point_list1[i]) - float(point_list2[i]))**2
        return math.sqrt(temp)

class CoordinateSystem(threading.Thread):
    coords = {}
    allConverged = {}
    semaphores = {}
    nodes = []
    avgPos = []
    def __init__(self, ID, neighbors, initialCoords, nodes):
        threading.Thread.__init__(self)
        self.id = ID
        self.neighbors = neighbors
        if ID == 1:
            CoordinateSystem.coords = initialCoords
            CoordinateSystem.nodes = nodes
            for i in range(len(nodes)):
                CoordinateSystem.semaphores[nodes[i]] = threading.Semaphore(0)
                CoordinateSystem.allConverged[nodes[i]] = False

    def nextDirection(self):
        position = CoordinateSystem.coords[self.id]
        neighborsCoords = []
        for neighbor in self.neighbors:
            neighborsCoords.append(CoordinateSystem.coords[neighbor])
        vecList = []
        for neighbor in neighborsCoords:
            vecList.append(unitVector(vectorFormation(position, neighbor)))
        return unitVector(vectorAddition(vecList))

    def updateMyPosition(self, nextMove):
        for dim in range(len(nextMove)):
            CoordinateSystem.coords[self.id][dim] += nextMove[dim]

    def updateDistance(self):
        flag = True
        current_pos = CoordinateSystem.coords[self.id]
        for neighbor in self.neighbors:
            if (distance(current_pos, CoordinateSystem.coords[neighbor])) > 2.0:
                flag = False
                break
        CoordinateSystem.allConverged[self.id] = flag

    def hasConverged(self):
        result = True
        for key in CoordinateSystem.allConverged.keys():
            if CoordinateSystem.allConverged[key] == False:
                result = False
                break
        return result

    def avgPosition(self):
        average = [0.0]*len(CoordinateSystem.coords[self.id])
        for node in CoordinateSystem.nodes:
            position = CoordinateSystem.coords[node]
            for dim in range(len(average)):
                average[dim] += position[dim]
        numNodes = float(len(CoordinateSystem.nodes))
        for dim in range(len(average)):
            average[dim] /= numNodes
        CoordinateSystem.avgPos = average

    def run(self):
        iter = 0
        while not self.hasConverged():
            iter += 1
            if (self.id == 1):
                print("Round {0}".format(iter))
            #Sync for all to begin
            CoordinateSystem.semaphores[self.id].release()
            for node in CoordinateSystem.nodes:
                CoordinateSystem.semaphores[node].acquire()
                CoordinateSystem.semaphores[node].release()
            
            nextMove = self.nextDirection()
            
            CoordinateSystem.semaphores[self.id].acquire()
            #Sync for all to begin updating coordinates
            CoordinateSystem.semaphores[self.id].release()
            for node in CoordinateSystem.nodes:
                CoordinateSystem.semaphores[node].acquire()
                CoordinateSystem.semaphores[node].release()
            
            self.updateMyPosition(nextMove)

            CoordinateSystem.semaphores[self.id].acquire()
            #Sync for all to have finished updating coordinates
            CoordinateSystem.semaphores[self.id].release()
            for node in CoordinateSystem.nodes:
                CoordinateSystem.semaphores[node].acquire()
                CoordinateSystem.semaphores[node].release()
            
            self.updateDistance()

            CoordinateSystem.semaphores[self.id].acquire()
            #Sync for all to have finished updating distances
            CoordinateSystem.semaphores[self.id].release()
            for node in CoordinateSystem.nodes:
                CoordinateSystem.semaphores[node].acquire()
                CoordinateSystem.semaphores[node].release()
            #...
            CoordinateSystem.semaphores[self.id].acquire()
        CoordinateSystem.semaphores[self.id].release()
        if (self.id == 1):
            self.avgPosition()

def avgPosition(coords, nodes):
    average = [0.0]*len(coords[nodes[0]])
    for node in nodes:
        position = coords[node]
        for dim in range(len(average)):
            average[dim] += position[dim]
    numNodes = float(len(nodes))
    for dim in range(len(average)):
        average[dim] /= numNodes
    return average

def graphConv(filename):
    graph = {}
    size = 0
    with open(filename, 'r') as file:
        for line in file:
            contents = line.split()
            if contents[0] == 'p':
                size = int(contents[2])
                break
    for i in range(1, size + 1):
        graph[i] = []
    with open(filename, 'r') as file:
        for line in file:
            edge = line.split()
            if edge[0] == 'e':
                e1 = int(edge[1])
                e2 = int(edge[2])
                graph[e1].append(e2)
                graph[e2].append(e1)
    return graph

def vectorFormation(startpoint, endpoint):
    vectorResult = []
    for dimension in range(len(endpoint)):
        vectorResult.append(float(endpoint[dimension]) - float(startpoint[dimension]))
    return vectorResult

def vectorAddition(vector_list):
    result = []
    for dimension in range(len(vector_list[0])):
        temp = 0.0
        for vector in vector_list:
            temp += float(vector[dimension])
        result.append(temp)
    return result

def vectorNegation(vector_list):
    result = []
    for dimension in range(len(vector_list[0])):
        temp = 0.0
        for vector in vector_list:
            temp -= float(vector[dimension])
        result.append(temp)
    return result

def vectorSubtraction(subXFromThis, X):
    for dimension in range(len(subXFromThis)):
        subXFromThis[dimension] -= X[dimension]
    return subXFromThis

def vectorMagnitude(vector):
    temp = 0.0
    for dimension in range(len(vector)):
        temp += float(vector[dimension])**2
    return math.sqrt(temp)

def unitVector(vector):
    magnitude = vectorMagnitude(vector)
    unit = []
    for dimension in range(len(vector)):
        unit.append(float(vector[dimension])/magnitude)
    return unit

def dot(A, B):
    dotScalar = 0.0
    for i in range(len(A)):
        dotScalar += A[i]*B[i]
    return dotScalar

def vectorProj(Aon, toB):
    toUnitB = unitVector(toB)
    dotScalar = dot(Aon, toUnitB)
    for i in range(len(toUnitB)):
        toUnitB[i] *= dotScalar
    return toUnitB

def equidistant_vectors(N, spacing):
    dims = N-1
    coords = {}
    first = [0.0]*dims
    second = [float(spacing)] + [0.0]*(dims-1)
    third = [float(spacing/2.0), float(spacing/2.0)*math.sqrt(3)] + [0.0]*(dims-2)
    coords[1] = first
    coords[2] = second
    coords[3] = third
    #initialize N equidistant vectors
    for i in range(4, N+1):
        node_coord = [0.0]*dims
        for dimension in range(0, i-2):
            temp = 0.0
            for prevNode in range(1, i):
                temp += coords[prevNode][dimension]
            temp /= float(i-1)
            node_coord[dimension] = temp
        temp = 0.0
        for prevDims in range(0, i-2):
            temp += (node_coord[prevDims]**2)
        last_dim_solution = math.sqrt(float(spacing)**2 - temp)
        node_coord[i-2] = last_dim_solution
        coords[i] = node_coord
    return coords

def convergence(coordinates, adjacency, space, onResume = False, resumedCoords = {}):
    start = time.time()
    nodes = adjacency.keys()
    distances = {}
    if onResume:
        coords = resumedCoords
        for i in range(len(nodes) - 1):
            lesser = nodes[i]
            distances[lesser] = {}
            for j in range(i+1, len(nodes)):
                greater = nodes[j]
                distances[lesser][greater] = [distance(resumedCoords[lesser], resumedCoords[nodes[j]]), False]
                if distances[lesser][greater][0] < 2.0:
                    distances[lesser][greater][1] = True
    else:
        coords = coordinates
        for i in range(len(nodes)-1):
            lesser = nodes[i]
            distances[lesser] = {}
            for j in range(i+1, len(nodes)):
                distances[lesser][nodes[j]] = [space, False]

    nodeThreads = []
    nodeThreads.append(CoordinateSystem(nodes[0], adjacency[nodes[0]], coords, nodes))
    nodeThreads[0].start()
    for node in range(1, len(nodes)):
        newThread = CoordinateSystem(nodes[node], adjacency[nodes[node]], coords, nodes)
        newThread.start()
        nodeThreads.append(newThread)

    for process in nodeThreads:
        process.join()
        
    print("Time to Converge: {0}".format(time.time() - start))
    return CoordinateSystem.avgPos

def main(filename, origSize, sample = False, sampleGraph = {}, onResume = False, resumeCoords = {}):
    graph = {}
    if sample:
        graph = sampleGraph
    else:
        graph = graphConv(filename)

    if True:
        spacing = 5000
        coords = equidistant_vectors(origSize, spacing)
        coordsCopy = {}
        for key in coords.keys():
            position = []
            for dimension in coords[key]:
                position.append(dimension)
            coordsCopy[key] = position
        
        cPoint = convergence(coordsCopy, graph, spacing, onResume, resumeCoords)
        print(cPoint)

        distResults = []
        for node in graph.keys():
            distResults.append((node, distance(cPoint, coords[node])))
        distResults = sorted(distResults, key=itemgetter(1), reverse=True)

        print(distResults)
        return distResults[0][0]

#1 Iteration took 0.6 sec
#Size 4 took 3868 simulated rounds
graph1 = {1:[2,3,4], 2:[1,3], 3:[1,2], 4:[1]}
#1 Iteration took 1.74 sec
#Size 7 took 3440 simulated rounds
graph2 = {1:[2,3,4,6], 2:[1,3,5,7], 3:[1,2,4,5], 4:[1,3,5,6,7], 5:[2,3,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#1 Iteration took 1.3 sec
graph2ver2 = {1:[2,4,6], 2:[1,5,7], 4:[1,5,6,7], 5:[2,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#1 Iteration took 1.0 sec
graph2ver3 = {1:[4,6], 4:[1,5,6,7], 5:[4,6,7], 6:[1,4,5,7], 7: [4,5,6]}
#1 Iteration took 0.6 sec
graph2ver4 = {4:[5,6,7], 5:[4,6,7], 6:[4,5,7], 7: [4,5,6]}

graph = graph2
#main("c125.txt", 7, True, graph)

#'''
#1 Iteration took ___ sec
DIMACS1 = "c125.txt"
#1 Iteration took ___ sec
DIMACS2 = "c250.txt"

filename = DIMACS1
main(filename, 125)
#'''
