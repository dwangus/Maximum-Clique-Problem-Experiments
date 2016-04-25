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
#clique = testSample(sampleGraph)
#size = len(clique)
#print("Returned Clique Size: {0}".format(size))
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
#graphRedux = {1: [2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 2: [1, 5, 17, 18, 22, 31, 33, 45, 48, 52, 65, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 5: [1, 2, 13, 17, 18, 31, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 13: [1, 5, 17, 18, 22, 31, 33, 45, 52, 55, 66, 67, 68, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 17: [1, 2, 5, 13, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 18: [1, 2, 5, 13, 17, 22, 31, 33, 45, 48, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 111, 115, 120, 121, 123], 22: [1, 2, 13, 17, 18, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 31: [1, 2, 5, 13, 17, 18, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 33: [1, 2, 13, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 68, 93, 96, 98, 101, 103, 104, 111, 120, 121, 123], 45: [1, 2, 5, 13, 17, 18, 22, 31, 33, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 96, 98, 101, 103, 104, 115, 120, 121, 123], 48: [1, 2, 5, 17, 18, 22, 31, 33, 45, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 52: [1, 2, 5, 13, 17, 22, 31, 33, 45, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 55: [1, 5, 13, 17, 18, 22, 31, 33, 45, 52, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 65: [1, 2, 5, 17, 22, 31, 33, 45, 48, 52, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 123], 66: [1, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 120, 121], 67: [2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 120, 121, 123], 68: [1, 2, 5, 13, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 71, 77, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 71: [1, 2, 5, 17, 18, 22, 31, 45, 48, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 77: [1, 2, 5, 13, 17, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 68, 71, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 89: [1, 2, 5, 13, 17, 18, 22, 45, 48, 52, 55, 65, 66, 67, 71, 77, 93, 98, 101, 103, 104, 111, 115, 120, 123], 93: [1, 2, 5, 13, 17, 18, 22, 31, 33, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 96: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 93, 98, 103, 104, 111, 120, 121, 123], 98: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 101, 103, 104, 111, 115, 120, 121], 101: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 98, 103, 111, 115, 120, 121, 123], 103: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 52, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 104: [1, 2, 5, 13, 17, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 103, 111, 115, 120, 121, 123], 111: [1, 2, 5, 13, 18, 22, 33, 48, 52, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 115: [1, 2, 5, 13, 17, 18, 22, 31, 45, 48, 52, 55, 65, 68, 71, 77, 89, 93, 98, 101, 103, 104, 111, 120, 121, 123], 120: [1, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 65, 66, 67, 71, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 121: [1, 2, 5, 17, 18, 31, 33, 45, 48, 52, 55, 66, 67, 68, 71, 77, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 123: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 67, 68, 71, 77, 89, 93, 96, 101, 103, 104, 111, 115, 120, 121]}
#graphRedux = {9: [11, 19, 25, 29, 34, 40, 44, 49, 59, 60, 61, 79, 80, 83, 85, 92, 110, 122, 125], 11: [9, 19, 25, 29, 34, 40, 44, 49, 60, 61, 79, 80, 83, 85, 92, 110, 122, 125], 19: [9, 11, 25, 26, 29, 34, 40, 44, 49, 59, 79, 80, 85, 92, 110, 122, 125], 25: [9, 11, 19, 26, 29, 34, 40, 44, 49, 59, 60, 61, 79, 80, 83, 85, 92, 110, 122, 124, 125], 26: [19, 25, 29, 34, 40, 44, 49, 59, 60, 61, 80, 85, 92, 110, 122, 124, 125], 29: [9, 11, 19, 25, 26, 34, 40, 44, 49, 59, 60, 61, 79, 80, 83, 85, 92, 110, 122, 124, 125], 34: [9, 11, 19, 25, 26, 29, 40, 44, 49, 59, 60, 61, 79, 80, 83, 85, 92, 110, 122, 124, 125], 40: [9, 11, 19, 25, 26, 29, 34, 44, 49, 59, 60, 61, 79, 80, 83, 92, 110, 122, 124, 125], 44: [9, 11, 19, 25, 26, 29, 34, 40, 49, 59, 60, 61, 79, 80, 83, 85, 92, 110, 122, 125], 49: [9, 11, 19, 25, 26, 29, 34, 40, 44, 60, 61, 79, 80, 83, 85, 92, 110, 122, 124, 125], 59: [9, 19, 25, 26, 29, 34, 40, 44, 60, 61, 79, 80, 83, 85, 92, 110, 124, 125], 60: [9, 11, 25, 26, 29, 34, 40, 44, 49, 59, 61, 79, 80, 83, 85, 110, 122, 124, 125], 61: [9, 11, 25, 26, 29, 34, 40, 44, 49, 59, 60, 85, 92, 110, 122, 124, 125], 79: [9, 11, 19, 25, 29, 34, 40, 44, 49, 59, 60, 80, 83, 85, 92, 110, 122, 124, 125], 80: [9, 11, 19, 25, 26, 29, 34, 40, 44, 49, 59, 60, 79, 83, 85, 110, 122, 124, 125], 83: [9, 11, 25, 29, 34, 40, 44, 49, 59, 60, 79, 80, 85, 92, 110, 122, 124], 85: [9, 11, 19, 25, 26, 29, 34, 44, 49, 59, 60, 61, 79, 80, 83, 92, 110, 122, 125], 92: [9, 11, 19, 25, 26, 29, 34, 40, 44, 49, 59, 61, 79, 83, 85, 110, 122, 124, 125], 110: [9, 11, 19, 25, 26, 29, 34, 40, 44, 49, 59, 60, 61, 79, 80, 83, 85, 92, 122, 124, 125], 122: [9, 11, 19, 25, 26, 29, 34, 40, 44, 49, 60, 61, 79, 80, 83, 85, 92, 110, 124, 125], 124: [25, 26, 29, 34, 40, 49, 59, 60, 61, 79, 80, 83, 92, 110, 122, 125], 125: [9, 11, 19, 25, 26, 29, 34, 40, 44, 49, 59, 60, 61, 79, 80, 85, 92, 110, 122, 124]}
#graphRedux = {7: [9, 11, 25, 29, 34, 41, 44, 49, 59, 70, 80, 82, 99, 110, 117, 122], 9: [7, 11, 25, 29, 34, 41, 44, 49, 58, 59, 70, 80, 82, 99, 110, 117, 122], 11: [7, 9, 25, 29, 34, 41, 44, 49, 58, 70, 80, 82, 99, 110, 117, 122], 25: [7, 9, 11, 29, 34, 41, 44, 49, 58, 59, 70, 80, 82, 99, 110, 117, 122], 29: [7, 9, 11, 25, 34, 41, 44, 49, 58, 59, 70, 80, 82, 99, 110, 117, 122], 34: [7, 9, 11, 25, 29, 41, 44, 49, 59, 70, 80, 82, 99, 110, 117, 122], 41: [7, 9, 11, 25, 29, 34, 44, 58, 59, 80, 82, 99, 110, 117, 122], 44: [7, 9, 11, 25, 29, 34, 41, 49, 58, 59, 70, 80, 82, 99, 110, 117, 122], 49: [7, 9, 11, 25, 29, 34, 44, 58, 70, 80, 82, 99, 110, 117, 122], 58: [9, 11, 25, 29, 41, 44, 49, 59, 70, 82, 99, 110, 117], 59: [7, 9, 25, 29, 34, 41, 44, 58, 70, 80, 82, 99, 110, 117], 70: [7, 9, 11, 25, 29, 34, 44, 49, 58, 59, 80, 82, 99, 110, 117, 122], 80: [7, 9, 11, 25, 29, 34, 41, 44, 49, 59, 70, 82, 99, 110, 117, 122], 82: [7, 9, 11, 25, 29, 34, 41, 44, 49, 58, 59, 70, 80, 110, 117, 122], 99: [7, 9, 11, 25, 29, 34, 41, 44, 49, 58, 59, 70, 80, 110, 117, 122], 110: [7, 9, 11, 25, 29, 34, 41, 44, 49, 58, 59, 70, 80, 82, 99, 117, 122], 117: [7, 9, 11, 25, 29, 34, 41, 44, 49, 58, 59, 70, 80, 82, 99, 110, 122], 122: [7, 9, 11, 25, 29, 34, 41, 44, 49, 70, 80, 82, 99, 110, 117]}

def mapping(graph):
    size = len(graph.keys())
    mapNodes = {}
    i = 1
    for item in graph.keys():
        mapNodes[item] = i
        i += 1
    newMap = {}
    for key in graph.keys():
        temp = []
        neighbors = graph[key]
        for n in neighbors:
            temp.append(mapNodes[n])
        newMap[mapNodes[key]] = temp
    reverseMap = {}
    for key in mapNodes.keys():
        reverseMap[mapNodes[key]] = key
    return newMap, reverseMap
def reverseCliqueMap(mapkey, clique):
    temp = []
    for member in clique:
        temp.append(mapkey[member])
    return temp
def makeGraphCopy(graph):
    copy = {}
    for key in graph.keys():
        temp = []
        for n in graph[key]:
            temp.append(n)
        copy[key] = temp
    return copy
def delete(toDel, graph):
    for node in toDel:
        neighbors = graph[node]
        for n in neighbors:
            graph[n].remove(node)
        graph.pop(node, None)
    return graph
def neighborhoodOrderIterate(graph, ordering, x, OG, prevsols, base, uniqueT = False):
    sols = []
    for key in ordering.keys():
        temp = makeGraphCopy(graph)
        first = ordering[key][x]
        exile = []
        for node in temp.keys():
            if node != key and node != first:
                if node not in temp[key] or node not in temp[first]:
                    exile.append(node)
        temp = delete(exile, temp)
        H, mapkey = mapping(temp)
        clique = testSample(H)
        if len(clique) == 14:
            origMap = reverseCliqueMap(mapkey, clique)
            addedBase = sorted(base + origMap)
            if checkClique(addedBase, OG):
                sols.append(addedBase)
                print(origMap)
            else:
                print("Something went wrong")
            #print(origMap)
            #print("Returned Clique Size: {0}".format(len(clique)))
    for sol in sols:
        flag = False
        for solution in prevsols:
            for i in range(len(solution)):
                if sol[i] != solution[i]:
                    break
                elif i == (len(solution)-1):
                    flag = True
                    break
            if flag:
                break
        if not flag:
           print(sol)
    if uniqueT:
        unique = []
        for solution in prevsols:
            for item in solution:
                if item not in unique:
                    unique.append(item)
        unique = sorted(unique)
        print(unique)
        print(len(unique))
    return sols


solution1 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
solution2 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution3 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution4 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution5 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution16 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution6 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution7 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution8 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution9 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution10 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution11 = [1, 5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution12 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution13 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution14 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
solution15 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
solution17 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution18 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution19 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution20 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s21 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 111, 114, 115, 117, 121, 122, 123, 125]
s22 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s23 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s24 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
s25 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
s26 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s27 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s28 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s29 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s30 = [5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s31 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s32 = [1, 5, 7, 9, 11, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 121, 122, 125]
s33 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s34 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s35 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s36 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s37 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s38 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s39 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s40 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 40, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s41 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s42 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s43 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s44 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s45 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s46 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s47 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s48 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s49 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s50 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s51 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s52 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s53 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s54 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s55 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]


solutionDatabase = [solution1, solution2, solution3, solution4, solution5, solution6, solution7, solution8, solution9, solution10, solution11, solution12, solution13, solution14, solution15, solution16, solution17]
solutionDatabase += [s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55]

base = [29,44,83,85,79,60, 1,2,5,18,31,48,68,71,77,93,101,115,121,82]
#get ordering from test8.4_numpy.py, based on what base you use
ordering = {34: [80, 7, 9, 25, 54, 110, 117, 114, 123, 122, 11, 70, 41, 49, 59], 
123: [9, 25, 54, 110, 114, 117, 70, 7, 11, 34, 80, 49, 122, 41, 59, 58], 
70: [9, 25, 54, 117, 123, 110, 114, 49, 7, 34, 11, 80, 122, 59, 58], 
7: [34, 80, 54, 25, 117, 110, 114, 123, 9, 122, 70, 11, 49, 41, 59], 
9: [54, 25, 117, 110, 114, 123, 11, 70, 34, 7, 80, 49, 41, 122, 59, 58], 
11: [9, 25, 110, 114, 117, 123, 54, 122, 49, 34, 7, 70, 80, 41, 58], 
110: [9, 25, 123, 54, 114, 117, 11, 7, 34, 80, 70, 49, 41, 122, 59, 58], 
80: [7, 34, 9, 25, 54, 117, 110, 114, 123, 122, 70, 11, 41, 49, 59], 
49: [11, 70, 9, 25, 54, 110, 114, 117, 123, 122, 7, 34, 80, 58], 
114: [9, 25, 54, 110, 117, 123, 11, 80, 7, 34, 70, 41, 49, 122, 59, 58], 
117: [9, 25, 54, 114, 110, 123, 7, 11, 34, 70, 80, 41, 49, 122, 59, 58], 
54: [9, 25, 117, 114, 123, 110, 34, 7, 11, 70, 80, 41, 49, 122, 59, 58], 
41: [9, 25, 54, 117, 110, 114, 123, 34, 7, 80, 11, 59, 122, 58], 
25: [9, 54, 117, 110, 114, 123, 7, 11, 34, 70, 80, 49, 41, 122, 59, 58], 
58: [9, 25, 54, 117, 123, 110, 114, 11, 70, 59, 41, 49], 
59: [9, 117, 25, 54, 114, 110, 123, 41, 34, 80, 70, 7, 58], 
122: [7, 11, 34, 80, 25, 54, 9, 110, 114, 117, 123, 49, 70, 41], 
}
#similarly get graphRedux from test8.4_numpy.py, based on what base you use
graphRedux = {7: [9, 11, 25, 34, 41, 49, 54, 59, 70, 80, 110, 114, 117, 122, 123], 9: [7, 11, 25, 34, 41, 49, 54, 58, 59, 70, 80, 110, 114, 117, 122, 123], 11: [7, 9, 25, 34, 41, 49, 54, 58, 70, 80, 110, 114, 117, 122, 123], 25: [7, 9, 11, 34, 41, 49, 54, 58, 59, 70, 80, 110, 114, 117, 122, 123], 34: [7, 9, 11, 25, 41, 49, 54, 59, 70, 80, 110, 114, 117, 122, 123], 41: [7, 9, 11, 25, 34, 54, 58, 59, 80, 110, 114, 117, 122, 123], 49: [7, 9, 11, 25, 34, 54, 58, 70, 80, 110, 114, 117, 122, 123], 54: [7, 9, 11, 25, 34, 41, 49, 58, 59, 70, 80, 110, 114, 117, 122, 123], 58: [9, 11, 25, 41, 49, 54, 59, 70, 110, 114, 117, 123], 59: [7, 9, 25, 34, 41, 54, 58, 70, 80, 110, 114, 117, 123], 70: [7, 9, 11, 25, 34, 49, 54, 58, 59, 80, 110, 114, 117, 122, 123], 80: [7, 9, 11, 25, 34, 41, 49, 54, 59, 70, 110, 114, 117, 122, 123], 110: [7, 9, 11, 25, 34, 41, 49, 54, 58, 59, 70, 80, 114, 117, 122, 123], 114: [7, 9, 11, 25, 34, 41, 49, 54, 58, 59, 70, 80, 110, 117, 122, 123], 117: [7, 9, 11, 25, 34, 41, 49, 54, 58, 59, 70, 80, 110, 114, 122, 123], 122: [7, 9, 11, 25, 34, 41, 49, 54, 70, 80, 110, 114, 117, 123], 123: [7, 9, 11, 25, 34, 41, 49, 54, 58, 59, 70, 80, 110, 114, 117, 122]}

#neighborhoodOrderIterate(graphRedux, ordering, 2, graphConv("c125.txt"), solutionDatabase, base)


















































