import math
import time
from operator import itemgetter

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

def nonNeigh(originNode, graph):
    unneighbors = []
    neighborhood = graph[originNode]
    for key in graph.keys():
        if key != originNode:
            if key not in neighborhood:
                unneighbors.append(key)
    return unneighbors

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

def update(coords, delNodes, memoized):
    average = []
    for vector in memoized:
        if sum(vector) != 0.0:
            average.append(unitVector(vector))
    averageVec = vectorNegation(average)
    similarity = []
    for vector in memoized:
        sim = dot(vector, averageVec)
        similarity.append(sim)
    
    minIndex = 0
    for node in range(1, len(similarity)):
        if (node+1) not in delNodes:
            if similarity[node] < similarity[minIndex]:
                minIndex = node
            elif similarity[node] == similarity[minIndex]:
                continue
    nextToDel = minIndex + 1
    return nextToDel, similarity[minIndex]

def iterate(graphOrig, paused = False, fromLastTime = []):
    start_time = time.time()
    graph = {}
    for key in graphOrig.keys():
        copy = []
        for neighbor in graphOrig[key]:
            copy.append(neighbor)
        graph[key] = copy
    
    coords = equidistant_vectors(len(graph.keys()), 5000)
    toDel = []
    
    dims = len(coords[1])
    memoized = [[0.0 for i in range(dims)] for j in range(len(graph.keys()))]
    for point1 in coords.keys():
        cur_point = coords[point1]
        vecList = []
        neighbors = graph[point1]
        for point2 in neighbors:
            vecList.append(vectorFormation(cur_point, coords[point2]))
        if len(neighbors) == 0:
            continue
        else:
            memoized[point1 - 1] = vectorAddition(vecList)
    if paused:
        toDel += fromLastTime
        for deleting in toDel:
            cur_point = coords[deleting]
            for node in graph[deleting]:
                vectorToSubtract = vectorFormation(coords[node], cur_point)
                graph[node].remove(deleting)
                memoized[node-1] = vectorSubtraction(memoized[node-1], vectorToSubtract)
            graph[deleting] = []
            memoized[deleting-1] = [0.0]*dims

    ##################
    error = -10000.0
    i = 0
    errorMargin = -0.000001
    while error < errorMargin:
        nextToDel, newError = update(coords, toDel, memoized)
        if newError > errorMargin:
            break
        toDel.append(nextToDel)
        error = newError
        #print("Deleting: {0}".format(nextToDel))
        #print(newError)

        cur_point = coords[nextToDel]
        for node in graph[nextToDel]:
            vectorToSubtract = vectorFormation(coords[node], cur_point)
            graph[node].remove(nextToDel)
            memoized[node-1] = vectorSubtraction(memoized[node-1], vectorToSubtract)
        graph[nextToDel] = []
        memoized[nextToDel-1] = [0.0]*dims
    ##################

    clique = []
    for key in graphOrig.keys():
        if key not in toDel:
            clique.append(key)

    flag = True
    for member in range(len(clique)-1):
        neighbors = graphOrig[clique[member]]
        for other in range(member+1, len(clique)):
            if clique[other] not in neighbors:
                #print(clique[member])
                #print(clique[other])
                flag = False
    if flag:
        for node in graphOrig.keys():
            if node not in clique:
                neighbors = graphOrig[node]
                flag = True
                for member in clique:
                    if member not in neighbors:
                        flag = False
                        break
                if flag:
                    print("Adding: {0}".format(node))
                    clique.append(node)
        print("Extracted Clique: {0}".format(clique))
        print("Time Spent: {0}".format(time.time() - start_time))
        return clique
    else:
        print("Extracted Clique: {0}".format(clique))
        print("Time Spent: {0}".format(time.time() - start_time))
        print("Didn't work")
        return []

def main(filename):
    graph = graphConv(filename)
    return iterate(graph)
#'''
file3 = "c500.txt"
file1 = "c125.txt"
fileName = file1
clique = main(fileName)
size = len(clique)
print("Returned Clique Size: {0}".format(size))
#'''
'''
#solution1 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]

graph = graphConv("c125.txt")
#[25, 11, 70, 9, 7, 19, 110, 96*, 29, 67*, 54, 79, 80, 125, 122, 117, 34, 40, 44, 5, 2, 77, 71, 17, 24*, 31, 123, 82*, 18, 48, 45, 49, 121]
# -- [96, 67, 24, 82] vs. [1, 99, 101, 114, 115]
clique = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 54, 70, 71, 77, 79, 80, 99, 101, 110, 114, 117, 122, 123, 125, 49, 115, 121]
def checkClique(clique, graph):
    flag = True
    for member in range(len(clique)-1):
        neighbors = graph[clique[member]]
        for other in range(member+1, len(clique)):
            if clique[other] not in neighbors:
                flag = False
                break
    print(clique)
    print("Size of Clique: {0}".format(len(clique)))
    print(flag)

checkClique(clique, graph)

#solution2 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
'''
#for c500.txt, found a clique of size 45 in 185 seconds





















































