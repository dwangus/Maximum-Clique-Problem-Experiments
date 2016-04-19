import math
import time
from operator import itemgetter
import numpy as np

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
    return np.asarray(endpoint) - np.asarray(startpoint)
def vectorAddition(vector_list):
    vector = np.asarray(vector_list[0])
    for vec in range(1, len(vector_list)):
        vector += np.asarray(vector_list[vec])
    return vector
def vectorNegation(vector_list):
    result = []
    vector = np.asarray(vector_list[0])*(-1)
    for vec in range(1, len(vector_list)):
        vector -= np.asarray(vector_list[vec])
    return vector
def vectorSubtraction(subXFromThis, X):
    return np.asarray(subXFromThis) - np.asarray(X)
def unitVector(vector):
    return (vector / np.linalg.norm(vector))
def dot(A, B):
    return np.dot(A, B)
def vectorProj(Aon, toB):
    toUnitB = unitVector(toB)
    dotScalar = dot(Aon, toUnitB)
    return toUnitB*dotScalar

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
    '''print(similarity)
    minIndex = 0
    for node in range(1, len(similarity)):
        if (node+1) not in delNodes:
            if similarity[node] < similarity[minIndex]:
                minIndex = node
            elif similarity[node] == similarity[minIndex]:
                continue
    nextToDel = minIndex + 1'''
    simSort = []
    for node in range(len(similarity)):
        if (node+1) not in delNodes:
            simSort.append((node+1, similarity[node]))
    simSort = sorted(simSort, key=itemgetter(1))
    return simSort

def iterate(graphOrig, paused = False, fromLastTime = []):
    start_time = time.time()
    if disconnected(graphOrig):
        return []
    graph = {}
    for key in graphOrig.keys():
        copy = []
        for neighbor in graphOrig[key]:
            copy.append(neighbor)
        graph[key] = copy
    
    coords = equidistant_vectors(len(graph.keys()), 5000)
    toDel = []
    for key in graph.keys():
        if len(graph[key]) == 0:
            toDel.append(key)
    
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
    #flaggy = True
    while error < errorMargin:
        similar = update(coords, toDel, memoized)
        #if flaggy:
        #    print(similar)
        #    flaggy = False
        nextToDel = similar[0][0]
        newError = similar[0][1]
        
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
        clique += analyzeNeighborhood(graphOrig, findAdditional(graphOrig, clique))
        '''for node in graphOrig.keys():
            if node not in clique:
                neighbors = graphOrig[node]
                flag = True
                for member in clique:
                    if member not in neighbors:
                        flag = False
                        break
                if flag:
                    print("Adding: {0}".format(node))
                    clique.append(node)'''
        #print("Extracted Clique: {0}".format(clique))
        #print("Time Spent: {0}".format(time.time() - start_time))
        return clique
    else:
        #print("Extracted Clique: {0}".format(clique))
        #print("Time Spent: {0}".format(time.time() - start_time))
        #print("Didn't work")
        return []

def main(filename):
    graph = graphConv(filename)
    return iterate(graph)
def testSample(graph):
    return iterate(graph)
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

def findAdditional(graph, clique):
    additional = []
    for key in graph.keys():
        if key not in clique:
            neighbors = graph[key]
            flag = True
            for member in clique:
                if member not in neighbors:
                    flag = False
                    break
            if flag:
                additional.append(key)
    return additional
def checkClique(clique, graph):
    flag = True
    for member in range(len(clique)-1):
        neighbors = graph[clique[member]]
        for other in range(member+1, len(clique)):
            if clique[other] not in neighbors:
                flag = False
                break
    #print("Return_Set: {0}".format(clique))
    #print("Size of Return_Set: {0}".format(len(clique)))
    #print("Return_Set is a Clique? -- {0}".format(flag))
    return flag
def constructNeighborhood(graph, clique):
    new = {}
    keys = {}
    for member in range(len(clique)):
        keys[member+1] = clique[member]
        neighbors = []
        for other in range(len(clique)):
            if other != member:
                if clique[other] in graph[clique[member]]:
                    neighbors.append(other+1)
        new[member+1] = neighbors
    return new, keys
def analyzeNeighborhood(graph, clique):
    newGraph, keys = constructNeighborhood(graph, clique)
    temp = iterate(newGraph)
    newClique = []
    for node in temp:
        newClique.append(keys[node])
    if checkClique(newClique, graph):
        return newClique
    else:
        return []
def disconnected(graph):    
    for key in graph.keys():
        if len(graph[key]) > 0:
            return False
    return True
def optimizeSub(graph, closedClique):
    takeOut = []
    for i in range(len(closedClique)):
        new = []
        for j in range(len(closedClique)):
            if j != i:
                new.append(clique[j])
        additional = findAdditional(graph, new)
        cliqueOf = analyzeNeighborhood(graph, additional)
        #takeOut.append((closedClique[i], len(cliqueOf), len(additional)))
        takeOut.append((closedClique[i], len(cliqueOf), len(additional), cliqueOf))
    takeOut = sorted(takeOut, key=itemgetter(1), reverse=True)
    print(takeOut)
    print("Previous Size: {0}".format(len(closedClique)))
    closedClique.remove(takeOut[0][0])
    print("Resultant Size: {0}".format(len(closedClique) + takeOut[0][1]))
    closedClique += takeOut[0][3]
    return sorted(closedClique)
def optimizeAdd(graph, openClique):
    print("Previous Size: {0}".format(len(openClique)))
    addition = analyzeNeighborhood(graph, findAdditional(graph, openClique))
    openClique += addition
    print("Resultant Size: {0}".format(len(openClique)))
    return sorted(openClique)
    


#'''
file3 = "c500.txt"
file2 = "c250.txt"
file1 = "c125.txt"
fileName = file3
sampleGraph = {1: [3, 4], 2: [3, 4, 5], 3: [1, 2], 4: [1, 2, 5], 5: [2, 4]}
#clique = main(fileName)
clique = testSample(sampleGraph)
size = len(clique)
print("Returned Clique Size: {0}".format(size))
#'''

#sol1 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
#c125 = [1, 2, 5, 7, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 54, 69, 70, 71, 77, 79, 80, 99, 101, 110, 114, 117, 122, 123, 125, 115]
c125 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]

#for c250.txt, found a clique of size 38 in 4.5 seconds
#optimizing using recursive-analyze method, found a clique of size 40 in about 5.5 seconds
#for c500.txt, found a clique of size 45 in 32.5 seconds
#[22, 28, 46, 48, 52, 55, 58, 61, 74, 81, 94, 97, 108, 113, 120, 132, 138, 143, 148, 164, 182, 183, 188, 223, 236, 243, 244, 260, 280, 294, 336, 339, 374, 383, 385, 388, 391, 403, 407, 411, 431, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]
#optimizing using recursive-analyze method, found a clique of size 51 in about 35 seconds
c250 = [17, 37, 41, 50, 58, 63, 64, 72, 73, 84, 86, 90, 104, 105, 117, 134, 136, 143, 150, 159, 161, 173, 174, 175, 184, 190, 195, 197, 198, 202, 212, 221, 226, 230, 241, 242]
c250ver2 = [17, 37, 41, 50, 58, 63, 64, 72, 73, 84, 86, 104, 105, 117, 134, 136, 143, 150, 159, 161, 173, 174, 175, 184, 190, 195, 197, 198, 202, 212, 221, 226, 230, 241, 242]

#sol500v1 = [21, 22, 33, 40, 46, 61, 63, 87, 97, 110, 121, 122, 132, 137, 155, 179*, 181, 182, 186, 189, 193, 194, 203, 212, 223, 244, 248, 249, 253, 266, 280, 290, 294*, 310, 316, 319*, 327, 329, 340, 350, 351, 357, 373, 374, 375, 381, 390, 395, 404, 405, 411, 415, 463, 478, 490, 491, 497]
c500 = [27, 28, 46, 48, 52, 55, 61, 74, 81, 94, 108, 113, 120, 132, 133, 138, 143, 148, 153, 188, 244, 280, 294, 336, 383, 385, 386, 391, 403, 407, 411, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]
c500ver2 = [28, 46, 48, 52, 55, 61, 74, 81, 94, 108, 113, 120, 132, 133, 138, 143, 148, 153, 188, 244, 280, 294, 336, 383, 385, 391, 403, 407, 411, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]
c500ver3 = [28, 46, 48, 52, 55, 61, 74, 81, 94, 108, 113, 120, 132, 138, 143, 148, 153, 188, 244, 280, 294, 336, 383, 385, 391, 403, 407, 411, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]
c500ver4 = [28, 46, 48, 52, 55, 61, 74, 81, 94, 108, 113, 120, 132, 133, 138, 143, 148, 188, 244, 280, 294, 336, 383, 385, 391, 403, 407, 411, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]
c500ver5 = [28, 46, 48, 52, 55, 61, 74, 81, 94, 108, 113, 120, 132, 138, 143, 148, 188, 244, 280, 294, 336, 383, 385, 391, 403, 407, 411, 433, 442, 448, 451, 453, 455, 462, 468, 480, 493]

c500ver6 = [21, 22, 33, 40, 46, 61, 63, 87, 97, 110, 121, 122, 132, 137, 155, 181, 182, 186, 189, 193, 194, 203, 212, 223, 244, 248, 249, 253, 266, 280, 290, 310, 316, 327, 329, 340, 350, 351, 357, 373, 374, 375, 381, 390, 395, 404, 405, 411, 415, 463, 478, 490, 491, 497]

#graph = graphConv(fileName)
#clique = c500ver6

#print(optimizeAdd(graph, clique))




















































